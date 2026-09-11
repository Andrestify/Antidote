"""
Script di training "ufficiale" per AntiDote (immunizzazione bi-livello).

Questo file è la versione "da codebase" (eseguibile da riga di comando) dello stesso
identico algoritmo prototipato ed eseguito interattivamente nel notebook `Untitled.ipynb`.
Rispetto alla versione originale di questo file, sono stati corretti 5 bug reali che
impedivano al loop di funzionare (vedi i commenti "FIX #n" sparsi nel codice):

  FIX #1 - Gradienti "morti"/che perdono: durante la Fase 1 (adversary) i parametri del
           defender NON venivano congelati, quindi accumulavano gradiente inutilmente;
           e l'optimizer del defender non veniva mai azzerato durante la Fase 1, quindi
           al primo step della Fase 2 del blocco successivo si applicava un gradiente
           "vecchio" e sporco. Risolto congelando esplicitamente i parametri non
           coinvolti in ogni fase e azzerando (zero_grad) entrambi gli optimizer.
  FIX #2 - "Hook leak": il context manager `with act_cache:` veniva ri-aperto a ogni
           step, e `ActivationCache.__exit__` non fa nulla (è un no-op), quindi ad ogni
           step si registravano NUOVI forward hook sopra quelli già esistenti, senza mai
           rimuoverli. Risolto registrando gli hook una sola volta fuori dal loop e
           chiamando solo `clear_cache()` ad ogni step.
  FIX #3 - Preprocessing mancante / chiavi sbagliate: la Fase 2 leggeva le chiavi
           `harmful_batch['safe_input_ids']` / `['harmful_input_ids']`, che non esistono
           (il dataset ha solo `prompt`/`chosen`/`rejected`), e passava le stringhe grezze
           del batch benigno direttamente al modello. Risolto passando entrambi i batch
           attraverso `preprocess_for_dpo` / `preprocess_for_it` prima di usarli.
  FIX #4 - Orientamento chosen/rejected: nel dataset `chosen` = risposta dannosa e
           `rejected` = risposta sicura (vedi `dataset_generation.py`) -- esattamente
           l'orientamento che l'ADVERSARY vuole. La loss di sicurezza del DEFENDER ha
           bisogno dell'orientamento opposto; il vecchio codice faceva uno scambio (swap)
           che, per errore, tornava all'orientamento dell'adversary. Risolto costruendo
           direttamente il batch del defender con chosen/rejected scambiati rispetto al
           batch grezzo.
  FIX #5 - `Accelerator` con bf16 / gradient checkpointing multi-GPU: tutta roba pensata
           per GPU Ampere+ (vedi `preflight.py`), che non serve (e in parte non funziona)
           per un singolo processo su CPU o su una singola GPU. Rimossa in favore di un
           loop semplice a singolo processo, con un parametro `device` esplicito.

In più, viene ora usato un controllo esplicito (ripreso da `preflight.py`) che impedisce
il bug silenzioso "num_blocks == 0": se il dataset è troppo piccolo rispetto a `k_steps`,
il loop bi-livello non eseguirebbe NESSUNO step, e la funzione arriverebbe comunque alla
fine restituendo il modello base non modificato, senza nessun errore. Ora questo caso
alza un errore chiaro invece di fallire in silenzio.
"""

import argparse
import copy
import json

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_cache import ActivationCache
from adversary import Adversary
from loss import compute_dpo_loss, compute_kl_loss, compute_lm_loss
from peft_injection import apply_adversarial_wrappers, restore_original_modules
from preflight import derive_layer_configs
from utils import preprocess_for_dpo, preprocess_for_it


def train_tamper_resistant_model(
    model: PeftModel,
    adversary: nn.Module,
    tokenizer,
    harmful_dataloader,
    benign_dataloader,
    k_steps: int = 2000,
    epochs: int = 3,
    lr_defender: float = 3e-5,
    lr_adversary: float = 2e-7,
    max_length: int = 2048,
    device: torch.device = None,
):
    """
    Loop di training bi-livello (AntiDote) a singolo processo.

    L'idea centrale (vedi il paper AntiDote) è un gioco a due giocatori che si alternano
    a blocchi:

      FASE 1 - L'ADVERSARY (un piccolo hypernetwork) osserva le attivazioni interne del
               defender su un prompt dannoso e impara a generare una "patch" LoRA
               (cioè una piccola correzione ai pesi, a basso rango, vedi `adversary.py`)
               che, se iniettata nel modello, lo spinge a preferire la risposta dannosa
               (`chosen`, in questo dataset) rispetto a quella sicura (`rejected`).
               In pratica l'adversary simula un "malicious fine-tuner": qualcuno che
               prende il modello e cerca di ri-addestrarlo per renderlo dannoso.
      FASE 2 - Il DEFENDER (l'adapter LoRA chiamato "defender" dentro `model`) viene
               addestrato in modo che, ANCHE quando la patch dell'adversary è iniettata,
               il modello continui a preferire la risposta sicura. Allo stesso tempo,
               due loss di "retain" (mantenimento) lo tengono vicino al comportamento
               originale sulle istruzioni benigne, così il modello non "dimentica" come
               essere utile mentre diventa più sicuro.

    Args:
        model: il modello `PeftModel` con l'adapter LoRA "defender" già attaccato.
        adversary: la rete `Adversary` (hypernetwork) che genera le patch LoRA malevole.
        tokenizer: il tokenizer del modello.
        harmful_dataloader: DataLoader sul dataset di preferenze dannose/sicure
            (deve avere le chiavi 'prompt', 'chosen', 'rejected').
        benign_dataloader: DataLoader sul dataset di istruzioni benigne
            (deve avere le chiavi 'prompt', 'response').
        k_steps: numero di step di ottimizzazione per ciascuna fase, in ciascun blocco.
        epochs: quante volte ripetere l'intero programma di blocchi.
        lr_defender: learning rate dell'optimizer del defender (Fase 2).
        lr_adversary: learning rate dell'optimizer dell'adversary (Fase 1).
        max_length: lunghezza massima (in token) usata per troncare prompt+risposta.
        device: dispositivo torch su cui eseguire il training. Se None, usa la GPU se
            disponibile, altrimenti la CPU.

    Returns:
        Una tupla (hardened_model, adversary, history):
          - hardened_model: il modello base risultante, con l'adapter "defender" già
            fuso (merge) nei pesi originali -- un checkpoint HF "normale", senza più
            bisogno di PEFT per essere ricaricato.
          - adversary: la rete adversary allenata (utile per continuare/analizzare).
          - history: dizionario di liste con l'andamento delle 4 loss nel tempo, per
            fare i grafici diagnostici (vedi le celle "Metrics Plots" nel notebook).
    """
    # Se non specificato esplicitamente, usiamo la GPU se c'è, altrimenti la CPU.
    # Questo sostituisce l'uso di `accelerate.Accelerator`, che serviva solo per gestire
    # più GPU e la mixed precision bf16 -- inutile (e in parte rotto, vedi FIX #5) per un
    # singolo processo.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Spostiamo modello e adversary sul device scelto. `model` è un PeftModel (modello
    # base + adapter LoRA "defender"): spostarlo sposta sia i pesi base che l'adapter.
    model.to(device)
    adversary.to(device)

    # --- Modello di riferimento (reference model) ---
    # Serve da "ancora" per due cose:
    #   1) la loss DPO confronta le probabilità del modello in allenamento con quelle di
    #      questa copia "congelata", per non spostarsi troppo in fretta;
    #   2) la loss KL (retain) confronta le distribuzioni di probabilità dei due modelli
    #      sulle istruzioni benigne, per misurare quanto il defender si è "allontanato"
    #      dal comportamento originale.
    # `model.base_model` è il modello Hugging Face "nudo" sotto il wrapper PEFT: lo
    # copiamo con `copy.deepcopy` così le modifiche successive al defender non lo toccano.
    with torch.no_grad():
        ref_model = copy.deepcopy(model.base_model)
    ref_model.eval().to(device)
    # Congeliamo TUTTI i parametri del modello di riferimento: non lo alleniamo mai,
    # serve solo per calcolare log-probabilità "di confronto".
    for p in ref_model.parameters():
        p.requires_grad = False

    # Selezioniamo solo i parametri che appartengono all'adapter "defender" (identificati
    # dal fatto che la stringa "defender" compare nel loro nome completo, es.
    # "base_model.model.layers.0.self_attn.q_proj.lora_A.defender.weight"). Sono questi,
    # e solo questi, i pesi che il DEFENDER può modificare.
    defender_params = [p for n, p in model.named_parameters() if "defender" in n]

    # Due optimizer separati: uno per il defender (i pesi LoRA "defender" dentro `model`),
    # uno per l'adversary (la rete hypernetwork esterna). Sono aggiornati in fasi diverse
    # e non devono mai interferire tra loro.
    optimizer_d = AdamW(defender_params, lr=lr_defender)
    optimizer_a = AdamW(adversary.parameters(), lr=lr_adversary)

    # Nomi dei sotto-moduli su cui l'adversary "guarda" le attivazioni (le proiezioni di
    # attention query/key/value/output). Sono gli stessi layer su cui poi iniettiamo la
    # patch LoRA malevola generata dall'adversary.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # --- FIX #2: registriamo gli hook UNA SOLA VOLTA, fuori dal loop di training ---
    # Un "forward hook" è una funzione che PyTorch chiama automaticamente ogni volta che
    # un certo sotto-modulo completa il suo forward pass; qui la usiamo per "spiare" (e
    # salvare) l'input di q/k/v/o_proj, cioè le attivazioni che l'adversary userà come
    # contesto per generare la sua patch. Il codice originale ri-registrava gli hook a
    # ogni singolo step tramite `with act_cache:` (che chiama `register_hooks()` a ogni
    # `__enter__`, ma non li rimuove mai in `__exit__`): il numero di hook cresceva senza
    # limite, sprecando memoria e tempo. Qui li registriamo una volta e poi puliamo solo i
    # valori salvati (`clear_cache()`) ad ogni step.
    act_cache = ActivationCache(model, target_modules)
    act_cache.register_hooks()

    def make_adversarial_patch(input_ids, attention_mask):
        """
        Fa un forward pass "silenzioso" (senza calcolare gradiente, `torch.no_grad()`)
        sul defender per catturare le attivazioni nei layer target, poi le passa
        all'adversary per generare le matrici LoRA (U, V) della patch malevola.

        Ritorna un dizionario {nome_layer: (U, V)} pronto per essere iniettato dentro il
        modello da `apply_adversarial_wrappers` (vedi `peft_injection.py`).
        """
        # Puliamo i valori salvati dal giro precedente (le chiavi restano, i valori
        # tornano a None) prima di fare un nuovo forward pass.
        act_cache.clear_cache()

        with torch.no_grad():
            # Ci assicuriamo che sia attivo l'adapter "defender" (e non un altro adapter,
            # se in futuro ce ne fossero altri): l'adversary deve vedere le attivazioni
            # del defender così com'è ORA, prima di essere attaccato.
            model.set_adapter("defender")
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Per ciascun layer bersaglio, generiamo la coppia di matrici LoRA (U, V) a
            # partire dalle attivazioni catturate da quel layer.
            lora_weights = {}
            for layer_name, activations in act_cache.activations.items():
                # Il nome del "tipo" di layer (es. "q_proj") è l'ultimo pezzo del nome
                # completo (es. "...layers.3.self_attn.q_proj" -> "q_proj"): serve
                # all'adversary per scegliere la testa di proiezione giusta (vedi
                # `Adversary.forward` in `adversary.py`, che ha teste diverse per ogni
                # tipo/forma di layer).
                config_name = layer_name.split(".")[-1]
                U_gen, V_gen = adversary(activations.to(device), config_name)
                lora_weights[layer_name] = (U_gen, V_gen)
        return lora_weights

    # Iteratori "infiniti" sui due dataloader: quando un dataloader finisce (StopIteration)
    # lo ricreiamo da capo, così il training può fare più epoche/blocchi di quanti batch
    # ci siano nel dataset, senza fermarsi.
    harmful_iterator = iter(harmful_dataloader)
    benign_iterator = iter(benign_dataloader)

    # Dizionario in cui accumuliamo l'andamento delle 4 loss, step per step: utile per
    # fare grafici diagnostici dopo il training (vedi le celle "Metrics Plots" del
    # notebook) e per capire SE e QUANDO il training comincia ad andare storto.
    history = {
        "adversary_loss": [],
        "defender_safety_loss": [],
        "defender_lm_loss": [],
        "defender_kl_loss": [],
    }

    # Numero di blocchi (coppie Fase1+Fase2) per epoca, derivato dalla lunghezza del
    # dataloader dannoso: ogni blocco "consuma" k_steps batch in Fase 1 (anche se poi in
    # Fase 2 li rilegge tramite l'iteratore "infinito" sopra).
    num_total_steps = len(harmful_dataloader)
    num_blocks = num_total_steps // k_steps
    if num_blocks == 0:
        # Bug silenzioso descritto in `preflight.py` (`check_blocks`): se k_steps è più
        # grande del numero di batch disponibili, il ciclo `for block_idx in
        # range(num_blocks)` sotto non farebbe NESSUNA iterazione, e la funzione
        # arriverebbe comunque, senza errori, al merge finale -- restituendo il modello
        # di partenza, identico, come se fosse "hardened". Meglio fallire rumorosamente.
        raise ValueError(
            f"num_blocks calcolato come 0 (num_total_steps={num_total_steps}, "
            f"k_steps={k_steps}): il loop bi-livello non eseguirebbe alcuno step. "
            "Riduci k_steps o aumenta la dimensione del dataset. "
            "Vedi preflight.py --> check_blocks per stimare valori sensati."
        )

    try:
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

            for block_idx in range(num_blocks):
                # ================= FASE 1: TRAINING DELL'ADVERSARY =================
                print(f"--- Block {block_idx + 1}/{num_blocks}: training adversary per {k_steps} step ---")
                adversary.train()  # modalità training (attiva dropout ecc. nell'adversary)
                model.eval()  # il defender non deve aggiornarsi in questa fase

                # --- FIX #1 (parte 1): congeliamo ESPLICITAMENTE tutti i parametri del
                # modello (compreso il defender) durante la Fase 1. Senza questo, i pesi
                # del defender restano con requires_grad=True (valore di default quando
                # l'adapter LoRA viene creato) e accumulano gradiente ad ogni
                # `loss_a.backward()`, anche se poi `optimizer_a.step()` non li tocca:
                # quel gradiente "fantasma" resta lì, non azzerato, pronto a inquinare il
                # primo step della Fase 2 del blocco successivo.
                for _, p in model.named_parameters():
                    p.requires_grad = False
                # Solo l'adversary può allenarsi in questa fase.
                for p in adversary.parameters():
                    p.requires_grad = True

                for step in range(k_steps):
                    try:
                        harmful_batch = next(harmful_iterator)
                    except StopIteration:
                        # Il dataloader è finito: lo ricreiamo e ripartiamo dal primo batch.
                        harmful_iterator = iter(harmful_dataloader)
                        harmful_batch = next(harmful_iterator)

                    # NOTA: in questo dataset 'chosen' = risposta dannosa, 'rejected' =
                    # risposta sicura (vedi dataset_generation.py) -- quindi
                    # l'orientamento "grezzo" qui sotto è esattamente quello che vuole
                    # l'adversary: preferire chosen (la risposta dannosa).
                    #
                    # --- FIX #3: passiamo il batch grezzo (liste di stringhe prompt/
                    # chosen/rejected) attraverso `preprocess_for_dpo`, che si occupa di
                    # applicare il chat template, tokenizzare, e mascherare (con -100) la
                    # parte di prompt nelle label, cosicché la loss venga calcolata SOLO
                    # sulla risposta e non sul prompt.
                    adv_batch = preprocess_for_dpo(harmful_batch, tokenizer, max_length=max_length)
                    adv_batch = {k: v.to(device) for k, v in adv_batch.items()}

                    # Generiamo la patch adversariale a partire dalla risposta "chosen"
                    # (dannosa): l'adversary deve imparare a spingere il modello proprio
                    # verso quella risposta.
                    lora_weights_adv = make_adversarial_patch(
                        adv_batch["chosen_input_ids"], adv_batch["chosen_attention_mask"]
                    )
                    # Iniettiamo temporaneamente la patch nei layer target del modello,
                    # sostituendo i moduli originali con `AdversarialPeftWrapper` (vedi
                    # `peft_injection.py`), che aggiunge alla loro uscita normale un
                    # termine extra `x @ (V @ U)^T` calcolato dalla patch.
                    original_modules = apply_adversarial_wrappers(model, lora_weights_adv)

                    # Calcoliamo la loss DPO (qui in variante "ipo") del modello CON la
                    # patch iniettata, rispetto al modello di riferimento: questa loss è
                    # bassa quando il modello (patchato) preferisce fortemente "chosen"
                    # (dannoso) rispetto a "rejected" (sicuro) -- esattamente l'obiettivo
                    # dell'adversary.
                    loss_a = compute_dpo_loss(model, ref_model, adv_batch, device)

                    # Ordine standard: azzeriamo i gradienti PRIMA di backward, così non
                    # si accumulano con quelli del passo precedente.
                    optimizer_a.zero_grad()
                    loss_a.backward()
                    # Clip del gradiente: evita che un singolo step "esploda" i pesi
                    # dell'adversary se la loss ha un picco anomalo.
                    torch.nn.utils.clip_grad_norm_(adversary.parameters(), 1.0)
                    optimizer_a.step()

                    # Rimuoviamo la patch e ripristiniamo i moduli originali del modello:
                    # da qui in avanti il modello torna a comportarsi "normalmente" (senza
                    # l'attacco), finché non richiameremo di nuovo `apply_adversarial_wrappers`.
                    restore_original_modules(model, original_modules)

                    history["adversary_loss"].append(loss_a.item())
                    if (step + 1) % 5 == 0:
                        print(f"  [adv] step {step + 1}/{k_steps} loss={loss_a.item():.4f}")

                # ================= FASE 2: TRAINING DEL DEFENDER =================
                print(f"--- Block {block_idx + 1}/{num_blocks}: training defender per {k_steps} step ---")
                adversary.eval()  # l'adversary non deve aggiornarsi in questa fase
                model.train()  # il defender (LoRA "defender") si allena ora

                # --- FIX #1 (parte 2): congeliamo l'adversary e "scongeliamo" SOLO i
                # parametri il cui nome contiene "defender". Tutti gli altri parametri
                # del modello base restano congelati (come lo erano già, essendo pesi
                # pre-addestrati che non vogliamo toccare).
                for p in adversary.parameters():
                    p.requires_grad = False
                for n, p in model.named_parameters():
                    p.requires_grad = "defender" in n

                for step in range(k_steps):
                    try:
                        harmful_batch = next(harmful_iterator)
                        benign_batch_raw = next(benign_iterator)
                    except StopIteration:
                        harmful_iterator = iter(harmful_dataloader)
                        benign_iterator = iter(benign_dataloader)
                        harmful_batch = next(harmful_iterator)
                        benign_batch_raw = next(benign_iterator)

                    # --- FIX #3: come in Fase 1, dobbiamo preprocessare (tokenizzare +
                    # mascherare) i batch grezzi prima di poterli usare. Il codice
                    # originale saltava questo passaggio e provava a leggere chiavi che
                    # non esistono (`safe_input_ids`, `harmful_input_ids`), oppure a
                    # passare stringhe grezze al modello: qui usiamo le stesse funzioni
                    # di preprocessing della Fase 1 (per il batch dannoso/sicuro) e la
                    # funzione dedicata alle istruzioni benigne.
                    dpo_batch = preprocess_for_dpo(harmful_batch, tokenizer, max_length=max_length)
                    dpo_batch = {k: v.to(device) for k, v in dpo_batch.items()}
                    benign_batch = preprocess_for_it(benign_batch_raw, tokenizer, max_length=max_length)
                    benign_batch = {k: v.to(device) for k, v in benign_batch.items()}

                    # Anche in questa fase generiamo (e iniettiamo) la patch adversariale:
                    # il defender deve imparare a resistere all'attacco, non solo a
                    # comportarsi bene quando non è sotto attacco.
                    lora_weights_adv = make_adversarial_patch(
                        dpo_batch["chosen_input_ids"], dpo_batch["chosen_attention_mask"]
                    )
                    original_modules = apply_adversarial_wrappers(model, lora_weights_adv)

                    # --- FIX #4: obiettivo di sicurezza. Sotto la patch adversariale, il
                    # defender deve preferire la risposta SICURA. Ma in `dpo_batch`,
                    # "chosen" è ancora orientato come nel dataset grezzo, cioè verso la
                    # risposta DANNOSA (vedi nota sopra). Per il defender dobbiamo quindi
                    # scambiare chosen/rejected UNA SOLA VOLTA rispetto al batch grezzo:
                    # "chosen" per il defender = "rejected" nel dataset grezzo (la
                    # risposta sicura), e viceversa. Il codice originale costruiva prima
                    # un dizionario con chiavi inesistenti e poi applicava un secondo
                    # scambio, che nel risultato finale annullava il primo, riportando
                    # tutto all'orientamento sbagliato (quello dell'adversary).
                    defender_dpo_batch = {
                        "chosen_input_ids": dpo_batch["rejected_input_ids"],
                        "chosen_attention_mask": dpo_batch["rejected_attention_mask"],
                        "chosen_labels": dpo_batch["rejected_labels"],
                        "rejected_input_ids": dpo_batch["chosen_input_ids"],
                        "rejected_attention_mask": dpo_batch["chosen_attention_mask"],
                        "rejected_labels": dpo_batch["chosen_labels"],
                    }
                    loss_s = compute_dpo_loss(model, ref_model, defender_dpo_batch, device)
                    # Ripristiniamo i moduli originali PRIMA di calcolare le loss di
                    # retain qui sotto: quelle vanno calcolate sul modello "pulito", senza
                    # la patch adversariale (l'attacco simulato serve solo per la loss di
                    # sicurezza sopra).
                    restore_original_modules(model, original_modules)

                    # Obiettivo di "retain" (mantenimento delle capacità): due loss che
                    # penalizzano il defender se si allontana troppo dal comportamento
                    # originale sulle istruzioni benigne.
                    #   - loss_lm: normale loss di next-token-prediction (cross-entropy)
                    #     sulla risposta benigna -- il modello deve ancora saper generare
                    #     risposte utili e sensate.
                    #   - loss_kl: divergenza (KL, tramite cross-entropy) tra le
                    #     distribuzioni di probabilità del defender e del modello di
                    #     riferimento sugli stessi token benigni -- non solo la risposta
                    #     "più probabile" deve restare sensata, ma l'intera distribuzione
                    #     di probabilità non deve spostarsi troppo.
                    loss_lm = compute_lm_loss(model, benign_batch)
                    loss_kl = compute_kl_loss(model, ref_model, benign_batch)

                    # Loss totale del defender: somma pesata di sicurezza + le due loss di
                    # retain. I pesi (1.0 / 0.8 / 0.3) controllano il compromesso tra
                    # "diventare sicuro" e "restare utile".
                    total_loss_d = 1.0 * loss_s + 0.8 * loss_lm + 0.3 * loss_kl

                    optimizer_d.zero_grad()
                    total_loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(defender_params, 1.0)
                    optimizer_d.step()

                    history["defender_safety_loss"].append(loss_s.item())
                    history["defender_lm_loss"].append(loss_lm.item())
                    history["defender_kl_loss"].append(loss_kl.item())
                    if (step + 1) % 5 == 0:
                        print(
                            f"  [def] step {step + 1}/{k_steps} "
                            f"safety={loss_s.item():.4f} lm={loss_lm.item():.4f} kl={loss_kl.item():.4f}"
                        )
    finally:
        # Qualunque cosa succeda (anche un errore), rimuoviamo sempre gli hook registrati
        # sopra: altrimenti resterebbero attaccati al modello anche dopo che questa
        # funzione è terminata, rallentando ogni futuro forward pass fatto su `model`.
        act_cache.remove_hooks()

    # A fine training, "fondiamo" (merge) l'adapter LoRA "defender" nei pesi del modello
    # base: il risultato è un modello Hugging Face "normale" (niente più bisogno di PEFT
    # per ricaricarlo), pronto per essere salvato e usato per l'inferenza.
    model.eval()
    model.set_adapter("defender")
    hardened_model = model.merge_and_unload()
    print("\nAdapter fuso (merge) con successo. Ritorno il modello base 'hardened', l'adversary allenato e la history.")

    return hardened_model, adversary, history


# -----------------------------
# Argument Parser
# -----------------------------
def get_args():
    """Definisce e legge gli argomenti da riga di comando per lanciare il training."""
    parser = argparse.ArgumentParser(description="Train Tamper-Resistant Model (AntiDote)")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="ID del modello Hugging Face da caricare",
    )
    parser.add_argument(
        "--harmful_path",
        type=str,
        default="data/dpo_data.json",
        help="Percorso del dataset di preferenze dannose/sicure (JSON)",
    )
    parser.add_argument(
        "--it_path",
        type=str,
        default="data/instruction_tuning_data.json",
        help="Percorso del dataset di instruction-tuning benigno (JSON)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="result/hardened_model",
        help="Cartella dove salvare il modello 'hardened' (dopo il merge)",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per entrambi i dataloader")
    parser.add_argument("--k_steps", type=int, default=2000, help="Numero di step per fase, per blocco")
    parser.add_argument("--epochs", type=int, default=3, help="Numero di epoche (giri sull'intero programma di blocchi)")
    parser.add_argument("--lr_defender", type=float, default=3e-5, help="Learning rate dell'optimizer del defender")
    parser.add_argument("--lr_adversary", type=float, default=2e-7, help="Learning rate dell'optimizer dell'adversary")
    parser.add_argument("--max_length", type=int, default=2048, help="Lunghezza massima (in token) di prompt+risposta")
    parser.add_argument("--lora_r", type=int, default=8, help="Rango (r) delle matrici LoRA generate dall'adversary")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo torch (es. 'cuda', 'cpu'). Se omesso, si sceglie automaticamente.",
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = get_args()

    model_id = args.model_name
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Caricamento del modello base ---
    # `torch_dtype` in bf16 farebbe risparmiare memoria su GPU, ma non è supportato in
    # modo nativo su tutte le GPU (serve Ampere+, vedi `preflight.py --> check_bf16`) né
    # su CPU: usiamo float32, semplice e sempre corretto; se hai una GPU recente puoi
    # passare --device cuda e volendo cambiare qui il dtype manualmente.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float32,
        cache_dir="cache_dir",
        trust_remote_code=True,
    )

    # Questi moduli vengono salvati/allenati "per intero" (non con LoRA a basso rango):
    # sono l'embedding dei token e i layer di normalizzazione, che è meglio lasciare
    # liberi di adattarsi invece di vincolarli a un aggiornamento a basso rango.
    modules_to_save = ["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"]

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=modules_to_save,
    )

    # Attacchiamo l'adapter LoRA "defender" al modello base: da qui in avanti `model` è
    # un `PeftModel` che, quando l'adapter "defender" è attivo, calcola
    # output_base + LoRA_defender(input).
    model = PeftModel(base_model, peft_config, adapter_name="defender")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # --- Configurazione dei layer per l'adversary ---
    # Deriviamo automaticamente le dimensioni (in_features, out_features) di
    # q_proj/k_proj/v_proj/o_proj dalla configurazione del modello, invece di usare un
    # dizionario scritto a mano per una taglia di modello fissa (bug: il file originale
    # aveva `layer_configs_3b` hardcoded, sbagliato per qualunque modello diverso da un
    # 3B). Questa funzione (`derive_layer_configs`, definita in `preflight.py`) tiene
    # conto anche della Grouped-Query Attention, dove k_proj/v_proj hanno un numero di
    # "teste" diverso da q_proj/o_proj.
    layer_configs = derive_layer_configs(model_id)
    adversary = Adversary(r=args.lora_r, layer_configs=layer_configs)

    # -----------------------------
    # Caricamento dei dataset
    # -----------------------------
    with open(args.harmful_path) as f:
        harmful_ds = json.load(f)

    with open(args.it_path) as f:
        it_ds = json.load(f)

    # Nota: ogni elemento di questi JSON è un dizionario con chiavi stringa (es. 'prompt',
    # 'chosen', 'rejected'); il collate_fn di default di DataLoader trasforma
    # automaticamente una lista di questi dizionari in un dizionario di liste (es.
    # {'prompt': [p1, p2], 'chosen': [c1, c2], ...}), che è esattamente il formato che
    # `preprocess_for_dpo`/`preprocess_for_it` si aspettano.
    harmful_loader = DataLoader(harmful_ds, batch_size=args.batch_size, shuffle=True)
    it_loader = DataLoader(it_ds, batch_size=args.batch_size, shuffle=True)

    # -----------------------------
    # Training
    # -----------------------------
    hardened_model, trained_adversary, history = train_tamper_resistant_model(
        model,
        adversary,
        tokenizer,
        harmful_loader,
        it_loader,
        k_steps=args.k_steps,
        epochs=args.epochs,
        lr_defender=args.lr_defender,
        lr_adversary=args.lr_adversary,
        max_length=args.max_length,
        device=device,
    )

    # -----------------------------
    # Salvataggio del modello risultante
    # -----------------------------
    # `hardened_model` è già un modello Hugging Face "normale" (l'adapter è stato fuso
    # dentro `train_tamper_resistant_model`), quindi si salva in modo diretto.
    try:
        hardened_model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"Salvato il modello hardened + tokenizer in: {args.save_path}")
    except Exception as exc:
        print("ATTENZIONE: salvataggio automatico del modello fallito. Eccezione:", exc)
