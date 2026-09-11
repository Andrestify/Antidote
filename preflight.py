"""
Preflight (controlli "prima del volo") per la Week 1 -- riproduzione in piccola scala
di AntiDote.

Lancia questo script PRIMA di modificare training.py. Non allena nulla: controlla che
le tre cose che possono far fallire un run "in silenzio" (cioè senza errori, ma
producendo un risultato inutile o sbagliato) siano a posto.

    python preflight.py --model_name Qwen/Qwen2.5-0.5B-Instruct \
                        --n_harmful 1000 --batch_size 2 --k_steps 2000
"""

import argparse
import torch
from transformers import AutoConfig


def derive_layer_configs(model_name: str) -> dict:
    """
    Ritorna il dizionario {nome_modulo: (in_features, out_features)} che si aspetta
    `Adversary(...)` (vedi adversary.py), derivandolo direttamente dalla configurazione
    del modello invece di scriverlo a mano per una taglia di modello fissa. Sostituisce
    (ed evita) il vecchio `layer_configs_3b` hardcoded in training.py, che sarebbe
    sbagliato per qualsiasi modello diverso da quello per cui era stato scritto a mano.

    Attenzione a k_proj/v_proj: con la Grouped-Query Attention (GQA) l'output di questi
    due layer NON ha dimensione `hidden_size` come q_proj/o_proj, ma
    `num_key_value_heads * head_dim` (che può essere più piccola: è proprio l'idea della
    GQA, condividere key/value tra più teste di query per risparmiare memoria).
    """
    # Scarica (o legge dalla cache locale) solo il file di configurazione del modello,
    # senza caricarne i pesi: è molto più rapido che caricare l'intero modello solo per
    # leggere due numeri.
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    # Se il modello non usa Grouped-Query Attention, `num_key_value_heads` non esiste:
    # in quel caso il numero di teste key/value è uguale al numero di teste query.
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    # Alcuni modelli espongono `head_dim` esplicitamente; altrimenti si calcola dividendo
    # la dimensione nascosta per il numero di teste query.
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    kv_out = n_kv * head_dim

    return {
        "q_proj": (hidden, n_heads * head_dim),
        "o_proj": (n_heads * head_dim, hidden),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
    }


def check_bf16() -> None:
    """
    Controlla se la GPU disponibile (se c'è) supporta bf16 in modo NATIVO in hardware.
    bf16 (bfloat16) è un formato numerico a 16 bit usato per allenare più velocemente e
    con meno memoria rispetto a float32, mantenendo però lo stesso intervallo di valori
    rappresentabili di float32 (a differenza di float16, che ha un intervallo più
    piccolo e può più facilmente "esplodere" o andare a zero).
    """
    if not torch.cuda.is_available():
        print("  [!] No GPU visible: training is not feasible here.")
        return
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # bf16 nativo richiede una GPU Ampere (compute capability 8.0) o più recente.
    # NON fidarsi di `torch.cuda.is_bf16_supported()`: versioni recenti di PyTorch
    # ritornano True anche su GPU Turing (es. T4), che però lo emulano via software in
    # modo lento -- non è vero supporto nativo.
    native_bf16 = major >= 8
    print(f"  GPU: {name} (compute capability {major}.{minor}, {total_gb:.0f} GB)")
    if native_bf16:
        print("  Native bf16 -> mixed_precision='bf16' is fine.")
    else:
        print("  [!] NO native bf16 (pre-Ampere: T4, P100, V100).")
        print("      In training.py: Accelerator(mixed_precision='fp16').")
        print("      Beware that with fp16 the DPO loss can go NaN:")
        print("      if that happens, you need float32 logprobs or an L4/A100 GPU.")


def check_blocks(n_harmful: int, batch_size: int, k_steps: int, epochs: int = 3) -> None:
    """
    Controlla che il "programma a blocchi" del loop bi-livello (Fase 1 adversary + Fase 2
    defender, ripetute a blocchi) abbia effettivamente almeno un blocco da eseguire.

    Il loop bi-livello calcola `num_blocks = numero_di_batch // k_steps`. Se il dataset
    è troppo piccolo rispetto a k_steps, `num_blocks` diventa 0: in quel caso il ciclo
    `for block_idx in range(num_blocks)` non farebbe NESSUNA iterazione, e lo script
    arriverebbe comunque, senza nessun errore, fino al merge finale -- restituendo il
    modello di partenza, non modificato, come se fosse "hardened". È un fallimento
    silenzioso particolarmente insidioso perché non si vede nessun errore: si vede solo,
    dopo, che il modello non è cambiato per niente.
    """
    n_batches = n_harmful // batch_size
    num_blocks = n_batches // k_steps
    print(f"  examples={n_harmful}  batch_size={batch_size}  -> {n_batches} batches")
    print(f"  k_steps={k_steps} -> num_blocks = {num_blocks}")
    if num_blocks == 0:
        suggested = max(1, n_batches // 4)
        print("  [!!] num_blocks = 0: the bi-level loop does NOT run and the")
        print("       script reaches the merge without errors, returning an")
        print(f"       un-hardened model. Lower k_steps to ~{suggested}.")
    else:
        # Ogni blocco fa k_steps step di adversary E k_steps step di defender (per
        # questo il fattore "2" qui sotto): il totale è quindi epoche * blocchi * 2 * k_steps.
        total = epochs * num_blocks * 2 * k_steps
        print(f"  total optimization steps: {epochs} x {num_blocks} x 2 x {k_steps}"
              f" = {total:,}")
        if total > 5000:
            print("  [!] Too many for Colab. Reduce k_steps, epochs or dataset size.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n_harmful", type=int, default=15580)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--k_steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    print("\n[1] layer_configs derived from the model")
    for k, v in derive_layer_configs(args.model_name).items():
        print(f"  '{k}': {v},")
    print("  -> paste this dict in place of layer_configs_3b in training.py")

    print("\n[2] numerical precision")
    check_bf16()

    print("\n[3] k:k schedule")
    check_blocks(args.n_harmful, args.batch_size, args.k_steps, args.epochs)
    print()
