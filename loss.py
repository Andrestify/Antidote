from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss

# Questo file raccoglie tutte le funzioni/classi di "loss" (funzione di costo, cioè il
# numero che il training cerca di minimizzare) usate nel progetto: la loss standard di
# linguaggio (next-token prediction), la loss DPO (Direct Preference Optimization) per
# insegnare una preferenza tra due risposte, e la loss KL per il mantenimento del
# comportamento originale.


def log_p_loss(
    logits: torch.Tensor, labels: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """
    Calcola la loss di log-probabilità per un modello di linguaggio.

    Questa funzione calcola la cross-entropy tra i logit predetti e le label vere,
    tipicamente usata nei task di modellazione del linguaggio (cioè: "quanto è
    probabile, secondo il modello, il prossimo token corretto?").

    Args:
        logits (torch.Tensor): I logit predetti dal modello, tipicamente di forma
                               (batch_size, sequence_length, vocab_size). I "logit" sono
                               i punteggi grezzi (non normalizzati) prima della softmax:
                               più alto è il logit di un token, più probabile lo ritiene
                               il modello.
        labels (torch.Tensor): Le label vere, tipicamente di forma
                               (batch_size, sequence_length).
        vocab_size (int): La dimensione del vocabolario (numero di token possibili).

    Returns:
        torch.Tensor: La loss calcolata, come tensore scalare (un singolo numero).
    """
    # In un modello autoregressivo, la posizione i-esima dei logit predice il token alla
    # posizione i+1: per questo "shiftiamo" (spostiamo di uno) le label rispetto ai
    # logit. `[..., :-1, :]` scarta l'ultima posizione dei logit (non ha un token
    # "successivo" da confrontare), `[..., 1:]` scarta il primo token delle label (nessun
    # logit lo prevede).
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # CrossEntropyLoss (di default) calcola la media della cross-entropy su tutti i
    # token, ignorando automaticamente quelli con label == -100 (l'`ignore_index` di
    # default): sono esattamente i token di prompt/padding mascherati in utils.py.
    loss_fct = CrossEntropyLoss()
    # `CrossEntropyLoss` si aspetta un tensore 2D (N, vocab_size) di logit e un tensore
    # 1D (N,) di label: "appiattiamo" quindi le dimensioni batch e sequenza insieme.
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def log_1_minus_p_loss(
    _logits: torch.Tensor,
    _labels: torch.Tensor,
    vocab_size: int,
    threshold: float = -15.0,
) -> torch.Tensor:
    """
    Calcola la loss log(1 - P(label)) per un modello di linguaggio.

    A differenza di `log_p_loss` (che spinge il modello a massimizzare la probabilità
    del token corretto), questa funzione spinge il modello a MINIMIZZARE la probabilità
    del token "corretto" -- utile, ad esempio, per un obiettivo di "unlearning" (fare
    dimenticare al modello certi contenuti), anche se in questo progetto non risulta
    usata direttamente dal loop di training principale (vedi training.py).

    Un `threshold` evita di continuare a penalizzare token per cui il modello è già
    sufficientemente "incerto"/basso in probabilità: se log(P(label)) è già molto basso,
    non ha senso spingerlo ancora più giù (gradiente inutile/instabile).

    Args:
        _logits (torch.Tensor): I logit predetti dal modello, forma
                                (batch_size, sequence_length, vocab_size).
        _labels (torch.Tensor): Le label vere, forma (batch_size, sequence_length).
        vocab_size (int): Dimensione del vocabolario.
        threshold (float, optional): Soglia sotto la quale ignorare la loss. Default -15.0.

    Returns:
        torch.Tensor: La loss calcolata, come tensore scalare.
    """
    # Stesso shift autoregressivo di log_p_loss.
    logits = _logits[..., :-1, :].contiguous()
    labels = _labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    # log(sum(exp(logit))) su tutto il vocabolario, per ogni posizione: è il
    # "denominatore" (in scala logaritmica) della softmax, calcolato in modo
    # numericamente stabile con `logsumexp` invece di fare exp/sum/log separatamente.
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)

    # Le label con valore -100 sono "da ignorare" (mascherate): non possiamo usare -100
    # come indice per `torch.gather` (andrebbe fuori dai limiti dell'array), quindi le
    # sostituiamo temporaneamente con 0 solo per l'operazione di gather; il risultato per
    # quelle posizioni verrà comunque azzerato più sotto.
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0

    # Prendiamo, per ogni posizione, il logit corrispondente al token "label".
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(
        -1
    )

    # log(P(label)) = logit_del_label - log(somma degli exp di tutti i logit).
    log_p = logits_for_labels - log_sum_exp_all

    # Costruiamo una maschera "one-hot" (1 solo nella posizione del token label, 0 nelle
    # altre) per poter azzerare SOLO il logit del token corretto nel prossimo passaggio.
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)

    # Azzeriamo (portiamo a un valore enormemente negativo, cioè "probabilità ~0" dopo
    # l'exp) il logit del token corretto: così, ricalcolando logsumexp, otteniamo la
    # somma su TUTTI I TOKEN TRANNE quello corretto.
    masked_logits = logits * (1 - mask) + mask * (
        -1e10
    )  # Large negative value to approximate zero when exponentiated

    # log(somma degli exp dei logit, escluso il token corretto).
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)

    # log(1 - P(label)) = log(somma_senza_label) - log(somma_totale). È una identità
    # algebrica: P(non-label) = 1 - P(label) = somma_senza_label / somma_totale.
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all

    # I token mascherati (-100 in origine) non devono contribuire alla loss finale:
    # azzeriamo il loro contributo.
    ignored_values = labels == -100
    log_1_minus_p[ignored_values] = 0

    # Azzeriamo anche il contributo dei token per cui il modello è già "sufficientemente
    # sicuro" di NON prevedere il label (log_p sotto la soglia): evita gradienti inutili
    # su token già "vinti".
    below_threshold = log_p < threshold
    log_1_minus_p[below_threshold] = 0

    # Loss finale: media di -log(1-P(label)) sui soli token NON ignorati (dividiamo per
    # il numero di token validi, non per il totale, altrimenti i token mascherati
    # "diluirebbero" artificialmente la media).
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()

    return loss


def max_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Calcola la loss di entropia negativa media per i logit dati.

    L'entropia di una distribuzione di probabilità misura quanto è "incerta"/uniforme:
    alta entropia = il modello non ha una preferenza forte su nessun token; bassa
    entropia = il modello è molto sicuro su un token specifico. Minimizzare questa loss
    (che è l'entropia con il segno invertito) equivale a MASSIMIZZARE l'entropia, cioè a
    spingere il modello verso distribuzioni più uniformi/incerte (utile ad esempio per
    "confondere" deliberatamente un modello su certi input). Non risulta usata dal loop
    di training principale di questo progetto.

    Args:
        logits (torch.Tensor): Il tensore dei logit in ingresso.

    Returns:
        torch.Tensor: La loss di entropia negativa media.
    """
    softmax = F.softmax(logits, dim=-1)
    log_softmax = F.log_softmax(logits, dim=-1)
    # Entropia = -sum(p * log(p)) su tutto il vocabolario, per ogni posizione.
    entropy = torch.sum(-softmax * log_softmax, dim=-1).mean()
    # Restituiamo l'entropia con il segno cambiato: minimizzarla equivale a
    # massimizzare l'entropia vera.
    return entropy.mean() * -1


def _filter_dpo_inputs(
    inputs: Dict[str, torch.Tensor], chosen: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Estrae, da un batch DPO (che contiene sia i tensori "chosen_*" che "rejected_*"),
    solo quelli relativi a UNA delle due risposte, rinominandoli senza prefisso
    (input_ids/attention_mask/labels), così possono essere passati direttamente al
    modello come argomenti "normali".

    Args:
        inputs (Dict[str, torch.Tensor]): Dizionario con i tensori del batch DPO.
        chosen (bool, optional): Se True, estrae i tensori "chosen_*"; se False
                                 (default), estrae quelli "rejected_*".

    Returns:
        Dict[str, torch.Tensor]: Dizionario filtrato con solo input_ids/attention_mask/labels.
    """
    prefix = "chosen_" if chosen else "rejected_"
    # Se il batch non ha proprio il formato DPO (niente chiavi con questo prefisso),
    # lo ritorniamo inalterato: permette a questa funzione di essere usata anche su
    # batch "normali" (non DPO) senza rompersi.
    if f"{prefix}input_ids" not in inputs:
        return inputs
    return {
        "input_ids": inputs[f"{prefix}input_ids"],
        "attention_mask": inputs[f"{prefix}attention_mask"],
        "labels": inputs[f"{prefix}labels"],
    }


def _filter_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Filtra il dizionario di input mantenendo solo le chiavi che un modello Hugging Face
    si aspetta come argomenti (`input_ids`, `attention_mask`, `labels`): utile per
    "pulire" un dizionario che potrebbe contenere anche altre chiavi (es. metadati) prima
    di passarlo a `model(**inputs)`.

    Args:
        inputs (Dict[str, torch.Tensor]): Dizionario di tensori in input.

    Returns:
        Dict[str, torch.Tensor]: Dizionario filtrato con solo le chiavi specificate.
    """
    return {
        k: v
        for k, v in inputs.items()
        if k in ["input_ids", "attention_mask", "labels"]
    }


def compute_lm_loss(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
    chosen: bool = False,
) -> torch.Tensor:
    """
    Calcola l'obiettivo standard di "massima probabilità del prossimo token"
    (next-token prediction), cioè la normale loss di un modello di linguaggio.

    Questa funzione calcola la loss di log-probabilità per la predizione del prossimo
    token usando il modello e gli input dati. Supporta sia input "normali" sia input in
    formato DPO (nel qual caso, tramite `chosen`, si sceglie quale delle due risposte
    usare).

    Args:
        model (torch.nn.Module): Il modello da usare per la predizione.
        inputs (Dict[str, torch.Tensor]): I tensori di input per il modello.
        device (Optional[torch.device]): Device per il training distribuito. Default None.
        chosen (bool): Se gli input sono in formato DPO, indica quale risposta usare.
            Default False (cioè "rejected"). In questo progetto, `compute_lm_loss` viene
            sempre chiamata con un batch "normale" (non DPO, vedi `preprocess_for_it`)
            per la loss di retain del defender, quindi questo parametro resta al default.

    Returns:
        torch.Tensor: La loss di log-probabilità calcolata (scalare).
    """
    # `_filter_dpo_inputs` non fa nulla se `inputs` non è in formato DPO (vedi sopra):
    # qui serve solo a gestire anche il caso DPO in modo uniforme.
    outputs = model(
        **_filter_inputs(_filter_dpo_inputs(inputs, chosen)), output_hidden_states=False
    )
    return log_p_loss(
        outputs.logits,
        _filter_dpo_inputs(inputs, chosen).get("labels"),
        model.vocab_size,
    )

def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1
) -> torch.Tensor:
    """
    Allunga (fa il padding di) un tensore fino a una lunghezza specificata, lungo una
    dimensione data, riempiendo lo spazio in più con `pad_value`.

    Serve a rendere due tensori di lunghezza diversa (es. la sequenza "chosen" e quella
    "rejected" di un batch DPO, che possono avere lunghezze diverse) della stessa
    lunghezza, così da poterli concatenare in un unico tensore (vedi
    `DPOLoss.concatenated_inputs` sotto).

    Args:
        tensor (torch.Tensor): Il tensore da allungare.
        length (int): La lunghezza desiderata lungo la dimensione indicata.
        pad_value (Union[int, float]): Il valore da usare per il riempimento.
        dim (int, optional): La dimensione lungo cui fare il padding. Default -1 (l'ultima).

    Returns:
        torch.Tensor: Il tensore allungato alla lunghezza richiesta.
    """
    if tensor.size(dim) >= length:
        # Già lungo abbastanza (o troppo): non c'è nulla da fare.
        return tensor
    else:
        # Costruiamo un tensore di "riempimento" della forma giusta (stessa forma
        # dell'originale, tranne che nella dimensione `dim`, dove ha la lunghezza
        # mancante) e lo concateniamo in coda.
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

class DPOLoss(torch.nn.Module):
    """
    Modulo di loss per la Direct Preference Optimization (DPO): https://arxiv.org/abs/2305.18290.

    Basato sull'implementazione della libreria TRL di Hugging Face:
    https://github.com/huggingface/trl/blob/5d1deb1445828cfd0e947cb3a7925b1c03a283fc/trl/trainer/dpo_trainer.py#L844

    Idea di fondo del DPO: dato un prompt e due risposte (una preferita "chosen", una
    scartata "rejected"), si allena il modello ad aumentare la differenza tra
    "quanto il modello preferisce chosen rispetto a rejected" ORA rispetto a quanto la
    preferiva un modello di riferimento (di solito una copia congelata del modello
    prima del training). In questo modo il modello impara la preferenza SENZA bisogno
    di un vero e proprio modello di reward separato (come servirebbe con RLHF classico).

    Args:
        beta (float): Parametro di "temperatura" della loss DPO, tipicamente tra 0.1 e
            0.5. Più basso = il modello può allontanarsi di più dal modello di
            riferimento per soddisfare la preferenza. Default 0.1.
        label_smoothing (float): Parametro che codifica l'incertezza sulle etichette
            (cioè: quanto ci fidiamo che "chosen" sia davvero sempre la scelta giusta).
            Default 0.
        loss_type (str): Tipo di funzione di loss da usare. Deve essere uno tra
            ['sigmoid', 'hinge', 'ipo', 'kto_pair'].
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        device: torch.device = None,
    ):
        super(DPOLoss, self).__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.device = device

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Calcola la log-probabilità delle label date, sotto i logit dati.

        Args:
            logits: Logit del modello (non normalizzati). Forma:
                (batch_size, sequence_length, vocab_size)
            labels: Label per cui calcolare la log-probabilità. I token con valore
                label_pad_token_id vengono ignorati. Forma: (batch_size, sequence_length)
            average_log_prob: Se True, ritorna la log-probabilità MEDIA per token (non
                mascherato). Se False, ritorna la SOMMA delle log-probabilità dei token
                (non mascherati). ATTENZIONE: qui sotto (in `concatenated_forward`)
                questa funzione viene chiamata senza specificare `average_log_prob`,
                quindi si usa il default False -- cioè si confrontano SOMME di
                log-probabilità, non medie. Se due risposte hanno un numero diverso di
                token "reali" (non mascherati), la somma è influenzata anche dalla
                lunghezza, non solo dal "quanto piace" ciascun token al modello: è un
                dettaglio da tenere a mente quando si interpretano i valori assoluti di
                chosen_logps/rejected_logps (per il training DPO stesso non cambia nulla
                di sostanziale, perché la loss guarda sempre una DIFFERENZA calcolata con
                la stessa convenzione sia per il modello in training che per quello di
                riferimento).
            label_pad_token_id: L'ID usato per il padding delle label.
            is_encoder_decoder: Se il modello è di tipo encoder-decoder.

        Returns:
            Un tensore di forma (batch_size,) con la log-probabilità media/sommata delle
            label date.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            # Stesso shift autoregressivo visto in log_p_loss: il logit alla posizione i
            # predice il token alla posizione i+1.
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        # Maschera: True dove il token NON è mascherato (cioè fa parte della risposta
        # vera, non del prompt né del padding).
        loss_mask = labels != label_pad_token_id

        # I token mascherati (-100) non sono indici validi per `torch.gather`: li
        # sostituiamo temporaneamente con 0 (un token "fittizio", il cui valore verrà
        # comunque azzerato dalla maschera qui sotto).
        labels[labels == label_pad_token_id] = 0
        # Per ogni posizione, prendiamo la log-probabilità (log_softmax) del token
        # "label" corrispondente.
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            # Media sui soli token validi: sum(log_p * mask) / numero_di_token_validi.
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            # Somma sui soli token validi (i token mascherati contribuiscono 0 grazie
            # alla moltiplicazione per `loss_mask`).
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Esegue il modello dato sul batch dato, concatenando gli input "chosen" e
        "rejected" in un unico tensore.

        Lo facciamo per evitare due forward pass separati: uno solo, su un batch il
        doppio più grande, è più efficiente (soprattutto in configurazioni distribuite
        come FSDP).
        """
        # Concateniamo chosen e rejected lungo la dimensione di batch (dopo averli
        # allineati alla stessa lunghezza di sequenza con il padding, vedi
        # `concatenated_inputs` sotto).
        concatenated_batch = self.concatenated_inputs(
            batch,
            device=self.device,
        )
        # Quanti esempi "chosen" ci sono: ci servirà per separare di nuovo i risultati
        # dopo il forward pass concatenato.
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}
        # Un UNICO forward pass su [chosen; rejected] concatenati lungo la dimensione di
        # batch: le prime `len_chosen` righe del risultato riguardano "chosen", le
        # restanti "rejected".
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        # Log-probabilità (somma, vedi nota sopra su average_log_prob) per ciascuna
        # sequenza del batch concatenato.
        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        # Separiamo di nuovo i risultati "chosen" da quelli "rejected".
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    @staticmethod
    def static_concatenated_forward(
        model: torch.nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        device: torch.device,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Versione "statica" (senza bisogno di un'istanza di DPOLoss, quindi senza
        `self.device`, ma con `device` passato esplicitamente) di `concatenated_forward`,
        identica nella logica. Utile per essere chiamata senza dover prima costruire un
        oggetto DPOLoss."""
        concatenated_batch = DPOLoss.concatenated_inputs(
            batch,
            device=device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = DPOLoss.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    @staticmethod
    def compute_reference_log_probs(
        model, padded_batch: Dict, device: torch.device
    ) -> Dict:
        """Calcola le log-probabilità del modello di RIFERIMENTO per un singolo batch
        (già "paddato") di un dataset in formato DPO. Usata quando non si vogliono
        ricalcolare le logp di riferimento ad ogni step (es. pre-calcolandole una volta e
        salvandole nel batch stesso, sotto le chiavi 'reference_chosen_logps' /
        'reference_rejected_logps' -- vedi `dpo_loss_obj` sotto)."""

        # `torch.no_grad()`: il modello di riferimento non va mai allenato, quindi non
        # serve costruire il grafo computazionale per il backward (si risparmia memoria
        # e tempo).
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = DPOLoss.static_concatenated_forward(model, padded_batch, device)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatena gli input "chosen" e "rejected" in un unico tensore.

        Args:
            batch: Un batch di dati. Deve contenere le chiavi 'chosen_input_ids' e
                'rejected_input_ids', tensori di forma (batch_size, sequence_length).
            is_encoder_decoder: Se il modello è di tipo encoder-decoder.
            label_pad_token_id: L'ID di padding per le label.
            padding_value: Il valore di padding da usare per gli input_ids concatenati.
            device: Il device per gli input concatenati.

        Returns:
            Un dizionario con gli input concatenati sotto la chiave 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(
                batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1]
            )
        else:
            # Le sequenze "chosen" e "rejected" possono avere lunghezze diverse (perché
            # tokenizzate e paddate separatamente in preprocess_for_dpo): prima di
            # concatenarle, dobbiamo portarle tutte alla stessa lunghezza massima.
            max_length = max(
                batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
            )

        # Prima passata: gestiamo tutte le chiavi "chosen_*", allungandole (con
        # pad_to_length) fino a max_length e rinominandole in "concatenated_*".
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    # Le label si "paddano" con -100 (valore ignorato dalla loss).
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    # Gli input_ids si paddano con il valore di padding "normale" (di
                    # solito l'ID del token di padding/EOS).
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    # L'attention mask si paddano con 0 (token di riempimento, "non
                    # guardare qui").
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
        # Seconda passata: stessa cosa per le chiavi "rejected_*", ma questa volta
        # CONCATENANDO (torch.cat lungo dim=0, la dimensione di batch) al risultato
        # "chosen" già presente sotto la stessa chiave "concatenated_*": è qui che le due
        # metà si uniscono in un unico tensore più grande.
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            # Per i modelli encoder-decoder, il prompt viene passato all'encoder
            # separatamente e va semplicemente duplicato (repeat) per allinearsi alle due
            # metà (chosen/rejected) del batch concatenato.
            concatenated_batch["concatenated_input_ids"] = (
                batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola la loss DPO per un batch di log-probabilità del modello in training
        ("policy") e del modello di riferimento.

        Args:
            policy_chosen_logps (torch.Tensor): Log-probabilità del modello in training
                per le risposte "chosen". Forma: (batch_size)
            policy_rejected_logps (torch.Tensor): Log-probabilità del modello in training
                per le risposte "rejected". Forma: (batch_size)
            reference_chosen_logps (torch.Tensor): Log-probabilità del modello di
                riferimento per le risposte "chosen". Forma: (batch_size)
            reference_rejected_logps (torch.Tensor): Log-probabilità del modello di
                riferimento per le risposte "rejected". Forma: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Una tupla di tre tensori:
                - losses: la loss DPO per ciascun esempio del batch.
                - chosen_rewards: "reward" impliciti per le risposte chosen.
                - rejected_rewards: "reward" impliciti per le risposte rejected.

        Raises:
            ValueError: Se viene specificato un tipo di loss non riconosciuto.
        """
        # "Log-ratio" della policy: quanto il modello in training preferisce chosen
        # rispetto a rejected, in scala logaritmica (differenza di log-probabilità).
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        # Stessa cosa, ma per il modello di riferimento: rappresenta "quanto già
        # preferiva chosen" ancora PRIMA di questo training.
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # La differenza tra le due log-ratio misura QUANTO IN PIÙ (rispetto al modello di
        # riferimento) il modello in training preferisce ora chosen rispetto a rejected:
        # è proprio questa quantità che le varie loss DPO cercano di spingere verso
        # l'alto (cioè verso "preferire sempre di più chosen, rispetto a quanto già
        # faceva il modello di riferimento").
        logits = pi_logratios - ref_logratios

        # Il beta è un parametro di "temperatura" per la loss DPO, tipicamente in un
        # intervallo tra 0.1 e 0.5. Ignoriamo il modello di riferimento quando beta -> 0.
        # Il parametro label_smoothing codifica la nostra incertezza sulle etichette e
        # calcola una loss DPO più "conservativa".
        if self.loss_type == "sigmoid":
            # Variante originale del paper DPO: massimizza log-sigmoid(beta * logits),
            # cioè spinge "logits" (la differenza calcolata sopra) verso valori grandi e
            # positivi, con una curva a saturazione (sigmoid) che evita che la loss cerchi
            # di spingerla all'infinito.
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            # Variante "hinge" (come nelle SVM): nessuna penalità se `logits` supera già
            # una soglia (1/beta), penalità lineare altrimenti.
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # IPO (Identity Preference Optimization): penalizza la DISTANZA di `logits`
            # da un valore obiettivo fisso (1 / (2*beta)), invece di spingerlo
            # all'infinito come fa "sigmoid". In questo progetto è la variante
            # effettivamente usata (vedi compute_dpo_loss sotto): è generalmente più
            # stabile con dataset piccoli/rumorosi, perché non ha un incentivo a
            # "strapreferire" all'infinito una risposta.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # Variante ispirata a KTO: usa un termine di "KL" per-batch come riferimento,
            # invece di confrontare direttamente chosen vs rejected accoppiati.
            chosen_kl = (
                (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            )
            rejected_kl = (
                (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
            )

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_kl)),
                    1 - F.sigmoid(self.beta * (chosen_kl - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        # I "reward" impliciti del DPO: quanto, secondo il modello in training, è
        # cambiata (rispetto al riferimento) la preferenza per ciascuna risposta presa
        # singolarmente. Non partecipano al calcolo del gradiente (`.detach()`): sono
        # solo utili come metriche diagnostiche (es. per calcolare "reward_accuracies" in
        # dpo_loss_obj sotto).
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards


from contextlib import nullcontext


def dpo_loss_obj(
    policy_model: torch.nn.Module = None,
    ref_model: torch.nn.Module = None,
    batch: Dict[str, Union[List, torch.LongTensor]] = None,
    accelerator: Accelerator = None,
    gradient_accumulation_steps: int = None,
    scale: float = 1.0,
    backprop: bool = True,
):
    """
    Funzione "tutto in uno" che calcola la loss DPO (variante 'ipo') e, se richiesto,
    esegue direttamente il backward pass tramite un `Accelerator`.

    NOTA: questa funzione non risulta usata dal loop di training corrente
    (train_tamper_resistant_model in training.py usa invece `compute_dpo_loss` sotto,
    più semplice e senza dipendenza da un Accelerator esterno). Contiene inoltre un
    riferimento a una variabile `device` che non è né un parametro né definita
    localmente: se venisse chiamata nel ramo che la usa (quando il batch contiene già
    'reference_chosen_logps'/'reference_rejected_logps' pre-calcolate), solleverebbe un
    `NameError`. È lasciata così com'era; da non usare senza prima correggerla.
    """
    with torch.no_grad() if not backprop else nullcontext():
        dpo_loss = DPOLoss(beta=0.1, accelerator=accelerator, loss_type="ipo")
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = dpo_loss.concatenated_forward(policy_model, batch)

        # Se le log-probabilità di riferimento sono già state pre-calcolate e salvate nel
        # batch (per risparmiare un forward pass extra ad ogni step), le usiamo
        # direttamente; altrimenti le calcoliamo ora dal modello di riferimento.
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"].to(
                device
            )
            reference_rejected_logps = batch["reference_rejected_logps"].to(
                device
            )
        else:
            with torch.no_grad():
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = dpo_loss.concatenated_forward(ref_model, batch)

        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # "Accuracy" diagnostica: in che frazione degli esempi il reward implicito di
        # chosen supera quello di rejected (cioè il modello "ha capito" la preferenza).
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        loss = losses.mean() / gradient_accumulation_steps * scale
        if backprop:
            accelerator.backward(loss)
    return loss.item(), reward_accuracies.mean().item() / gradient_accumulation_steps

def compute_dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    device: torch.device,
) -> torch.Tensor:
    """
    Calcola la loss DPO e la ritorna come tensore differenziabile (cioè con il grafo
    computazionale ancora collegato, pronto per `.backward()`).
    Questa funzione NON esegue il backward pass: lo fa chi la chiama (vedi training.py).

    È la funzione effettivamente usata dal loop di training in training.py, sia per la
    loss dell'adversary (Fase 1) sia per la loss di sicurezza del defender (Fase 2) --
    con lo stesso identico codice, ma passando batch con orientamento chosen/rejected
    diverso (vedi i commenti "FIX #4" in training.py).
    """
    # Riusiamo la classe DPOLoss solo per i suoi metodi di supporto (concatenated_forward
    # e la logica di loss vera e propria), con beta=0.1 e variante "ipo" (vedi sopra il
    # perché "ipo" è una scelta più stabile per dataset piccoli).
    dpo_loss_util = DPOLoss(beta=0.1, device=device, loss_type="ipo")

    # 1. Log-probabilità dal modello in training (la "policy", cioè quello che stiamo
    # effettivamente allenando in questo momento -- può essere il defender O il modello
    # con la patch adversariale iniettata, secondo chi chiama questa funzione).
    (
        policy_chosen_logps,
        policy_rejected_logps,
        _, _,
    ) = dpo_loss_util.concatenated_forward(policy_model, batch)

    # 2. Log-probabilità dal modello di riferimento (congelato): `torch.no_grad()`
    # perché non dobbiamo mai propagare gradiente attraverso il modello di riferimento.
    with torch.no_grad():
        (
            reference_chosen_logps,
            reference_rejected_logps,
            _, _,
        ) = dpo_loss_util.concatenated_forward(ref_model, batch)

    # 3. Calcoliamo le loss per-esempio con il metodo `forward` di DPOLoss.
    losses, _, _ = dpo_loss_util(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )

    # 4. Ritorniamo la media delle loss del batch come singolo tensore scalare.
    return losses.mean()

def compute_kl_loss(model, ref_model, benign_batch):
    """
    Calcola una loss di divergenza KL (approssimata come cross-entropy diretta) tra la
    distribuzione di probabilità del modello in training e quella del modello di
    riferimento, sugli stessi token benigni. È una delle due loss di "retain" usate dal
    defender (insieme a `compute_lm_loss`): non basta che il TOKEN PIÙ PROBABILE resti lo
    stesso, vogliamo che l'INTERA distribuzione di probabilità (su tutto il vocabolario)
    non si allontani troppo da quella originale.
    """
    # 1. Logit dai due modelli. Il modello di riferimento è sempre "congelato"
    # (torch.no_grad(), nessun gradiente necessario); il modello in training invece deve
    # mantenere il grafo computazionale, perché è lui che va effettivamente allenato.
    with torch.no_grad():
        ref_outputs = ref_model(**benign_batch)
        ref_logits = ref_outputs.logits

    policy_outputs = model(**benign_batch)
    policy_logits = policy_outputs.logits

    # 2. Log-probabilità per la policy (in training) e probabilità "normali" (non log)
    # per il riferimento: ci serve la probabilità vera del riferimento per pesare la
    # cross-entropy al passo successivo.
    # Usiamo solo log_softmax per la policy, più stabile numericamente.
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)

    # 3. Cross-entropy "diretta" (forward): -sum(P_riferimento * log(Q_policy)). Questa
    # quantità è minima quando le due distribuzioni P e Q coincidono esattamente: è una
    # buona approssimazione pratica della vera divergenza KL, spesso usata perché più
    # stabile numericamente da calcolare rispetto alla formula KL "esatta".
    # Il segno negativo è gestito per convenzione (la loss si minimizza).
    cross_entropy = -(ref_probs * policy_log_probs).sum(dim=-1)

    # 4. Media sul batch e sulla lunghezza di sequenza.
    # Dobbiamo rispettare l'attention mask per non fare la media anche sui token di
    # padding (che non sono "vero" contenuto).
    mask = benign_batch['attention_mask']
    masked_loss = cross_entropy * mask

    # Media solo sui token NON mascherati (padding escluso).
    kl_loss = masked_loss.sum() / mask.sum()

    return kl_loss
