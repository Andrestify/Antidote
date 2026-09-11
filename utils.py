import torch
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Dict, List

# Questo file contiene le funzioni di "preprocessing": trasformano testo grezzo
# (stringhe: prompt, risposte) in tensori numerici che un modello Transformer può
# effettivamente elaborare. Sono i "mattoncini" usati sia dal training (training.py)
# sia dalla valutazione (nel notebook), quindi è importante capirli bene.


def preprocess_for_dpo(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    label_pad_token_id: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Prepara un batch di dati di preferenza (DPO: Direct Preference Optimization) per il training.

    "DPO" è una tecnica per insegnare a un modello a preferire una risposta ('chosen')
    rispetto a un'altra ('rejected'), per lo stesso prompt. Per farlo, la loss ha bisogno
    di sapere quanto è probabile, secondo il modello, ciascuna delle due risposte:
    questa funzione produce i tensori tokenizzati necessari a calcolare quella
    probabilità (log-probabilità) SOLO sulla parte di risposta, ignorando il prompt.

    Questa funzione prende un dizionario di liste (come lo produce
    `dataset.map(batched=True)` di Hugging Face `datasets`), dove ogni lista corrisponde
    alle stringhe 'prompt', 'chosen' e 'rejected'. Formatta e tokenizza i dati nei sei
    tensori richiesti dalla loss DPO custom (vedi loss.py).

    Args:
        examples (Dict[str, List[str]]): Un batch di esempi da un dataset Hugging Face.
                                         Chiavi richieste: 'prompt', 'chosen', 'rejected'.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        max_length (int): Lunghezza massima della sequenza per il troncamento.
        label_pad_token_id (int): L'ID usato per "mascherare" i token nelle label (cioè
            per dire alla loss "ignora questo token").

    Returns:
        Dict[str, torch.Tensor]: Un dizionario con i sei tensori tokenizzati richiesti.
    """
    # Alcuni modelli (soprattutto quelli solo-decoder, come Qwen/GPT) non hanno un token
    # di padding definito di default: se manca, usiamo il token di fine-sequenza (EOS)
    # anche come token di padding. È una convenzione comune, non un problema.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quanti esempi ci sono in questo batch (es. 2, se batch_size=2).
    batch_size = len(examples['prompt'])

    # --- 1. Formattiamo le risposte 'chosen' e 'rejected' con il chat template ---
    # Un "chat template" trasforma una lista di messaggi (ruolo + contenuto) nel formato
    # testuale esatto che il modello si aspetta in input (con i suoi token speciali, es.
    # <|im_start|>user ... <|im_end|>). Ogni famiglia di modelli ha il proprio formato:
    # `tokenizer.apply_chat_template` lo applica automaticamente, senza doverlo scrivere
    # a mano.
    chosen_full_texts = []
    rejected_full_texts = []
    prompt_only_texts = []

    for i in range(batch_size):
        prompt_text = examples['prompt'][i]
        chosen_text = examples['chosen'][i]
        rejected_text = examples['rejected'][i]

        # Testo completo (prompt + risposta "chosen"), come lo vedrebbe il modello se
        # generasse esattamente questa risposta.
        messages_chosen = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_text}
        ]
        chosen_full_texts.append(tokenizer.apply_chat_template(messages_chosen, tokenize=False))

        # Stessa cosa, ma con la risposta "rejected".
        messages_rejected = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rejected_text}
        ]
        rejected_full_texts.append(tokenizer.apply_chat_template(messages_rejected, tokenize=False))

        # Formattiamo anche il SOLO prompt (senza risposta), con
        # `add_generation_prompt=True`: questo aggiunge i token speciali che indicano
        # "ora tocca all'assistente rispondere" (es. "<|im_start|>assistant\n"), che sono
        # presenti anche nei due testi completi sopra. Ci serve per sapere ESATTAMENTE
        # quanti token occupa il prompt (compresi questi token speciali), così da poter
        # "tagliare via" (mascherare) solo il prompt e lasciare solo la risposta nelle
        # label più sotto.
        messages_prompt = [{"role": "user", "content": prompt_text}]
        prompt_only_texts.append(tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True))

    # --- 2. Tokenizziamo tutti i testi formattati ---
    # "Tokenizzare" significa trasformare il testo in una sequenza di numeri interi
    # (ID di token), l'unica cosa che un modello sa davvero elaborare.
    # `truncation=True, max_length=max_length`: se il testo è troppo lungo, viene tagliato.
    # `padding="longest"`: le sequenze più corte del batch vengono "allungate" con token
    # di padding fino alla lunghezza della sequenza più lunga DEL BATCH (non di
    # max_length), così tutte le sequenze nel batch hanno la stessa lunghezza e possono
    # stare nello stesso tensore.
    tokenized_chosen = tokenizer(
        chosen_full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest" # Pad to the longest sequence in this batch
    )
    tokenized_rejected = tokenizer(
        rejected_full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest"
    )
    # Tokenizziamo anche i soli prompt, ma SENZA padding: ci interessa solo la loro
    # lunghezza "vera" (numero di token), non un tensore allineato.
    tokenized_prompts = tokenizer(prompt_only_texts, truncation=True, max_length=max_length)

    # --- 3. Prepariamo il dizionario finale ---
    batch_dict = {}

    # --- 4. Creiamo le "label" mascherando la parte di prompt ---
    # Le "label" sono ciò che la loss usa come bersaglio: per ogni posizione, "qual è il
    # token corretto che il modello dovrebbe prevedere qui?". Partiamo da una copia degli
    # input_ids (gli stessi token in input), e poi sostituiamo con `label_pad_token_id`
    # (-100, un valore speciale riconosciuto da `CrossEntropyLoss` per dire "ignora
    # questo") tutti i token che appartengono al PROMPT: non vogliamo che il modello
    # venga valutato/allenato sul "prevedere" il prompt (che è dato, non generato), solo
    # sulla risposta.
    chosen_labels = torch.tensor(tokenized_chosen['input_ids']).clone()
    rejected_labels = torch.tensor(tokenized_rejected['input_ids']).clone()

    for i in range(batch_size):
        # Numero di token occupati dal prompt (uguale sia per la versione "chosen" che
        # per quella "rejected", perché il prompt è lo stesso).
        prompt_len = len(tokenized_prompts['input_ids'][i])

        # Mascheriamo (con -100) i primi `prompt_len` token: quella è la parte di prompt.
        chosen_labels[i, :prompt_len] = label_pad_token_id
        # Mascheriamo anche gli eventuali token di padding che si trovano DOPO la
        # risposta vera (attention_mask == 0 significa "token di riempimento, non reale").
        chosen_labels[i][tokenized_chosen['attention_mask'][i] == 0] = label_pad_token_id

        # Stessa identica logica per la versione "rejected".
        rejected_labels[i, :prompt_len] = label_pad_token_id
        rejected_labels[i][tokenized_rejected['attention_mask'][i] == 0] = label_pad_token_id

    # Mettiamo tutto nel dizionario di output: per ciascuna delle due risposte servono
    # tre tensori -- gli ID dei token in input, la maschera di attenzione (1 = token
    # reale, 0 = padding) e le label appena costruite.
    batch_dict["chosen_input_ids"] = torch.tensor(tokenized_chosen['input_ids'])
    batch_dict["chosen_attention_mask"] = torch.tensor(tokenized_chosen['attention_mask'])
    batch_dict["chosen_labels"] = chosen_labels

    batch_dict["rejected_input_ids"] = torch.tensor(tokenized_rejected['input_ids'])
    batch_dict["rejected_attention_mask"] = torch.tensor(tokenized_rejected['attention_mask'])
    batch_dict["rejected_labels"] = rejected_labels

    return batch_dict



def preprocess_for_it(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    label_pad_token_id: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Prepara un batch di dati di instruction-tuning (istruzioni benigne) per la loss
    di "retain" (mantenimento delle capacità originali).

    A differenza di `preprocess_for_dpo`, qui c'è UNA SOLA risposta per prompt (non una
    coppia chosen/rejected): l'obiettivo non è "preferire" una risposta rispetto a
    un'altra, ma semplicemente "continuare a saper generare risposte utili e sensate",
    esattamente come faceva il modello prima di essere modificato.

    Questa funzione prende un dizionario di liste (come lo produce
    `dataset.map(batched=True)`), dove ogni lista corrisponde alle stringhe 'prompt' e
    'response'. Formatta e tokenizza i dati nei tre tensori richiesti dalla funzione di
    loss `obj_standard_max_next_token` (in loss.py, tramite `compute_lm_loss`).

    Args:
        examples (Dict[str, List[str]]): Un batch di esempi da un dataset Hugging Face.
                                         Chiavi richieste: 'prompt', 'response'.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        max_length (int): Lunghezza massima della sequenza per il troncamento.
        label_pad_token_id (int): L'ID usato per mascherare i token nelle label.

    Returns:
        Dict[str, torch.Tensor]: Un dizionario con 'input_ids', 'attention_mask' e 'labels'.
    """
    # Come in preprocess_for_dpo: se il tokenizer non ha un pad token, usiamo l'EOS.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = len(examples['prompt'])
    full_texts = []
    prompt_only_texts = []

    # Iteriamo in parallelo su prompt e risposta (zip li accoppia elemento per elemento).
    for prompt, response in zip(examples['prompt'], examples['response']):
        # Formattiamo la conversazione completa (prompt + risposta) con il chat template.
        messages_complete = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_texts.append(tokenizer.apply_chat_template(messages_complete, tokenize=False))

        # Formattiamo anche il solo prompt, per sapere quanti token occupa (stessa
        # logica di preprocess_for_dpo: serve solo a calcolare `prompt_len` più sotto).
        messages_prompt = [{"role": "user", "content": prompt}]
        prompt_only_texts.append(tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True))

    # Tokenizziamo le conversazioni complete, chiedendo direttamente tensori PyTorch
    # (`return_tensors="pt"`) invece di semplici liste di interi come sopra: qui è solo
    # comodo farlo subito perché non serve nessuna elaborazione intermedia sulle liste.
    tokenized_complete = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt"
    )

    # Tokenizziamo i soli prompt, senza padding (`padding=False`): ci interessa solo la
    # lunghezza vera di ciascuno, non un tensore allineato.
    tokenized_prompts = tokenizer(
        prompt_only_texts,
        truncation=True,
        max_length=max_length,
        padding=False # More efficient
    )

    # Le label partono come copia degli input_ids...
    labels = tokenized_complete.input_ids.clone()

    for i in range(batch_size):
        prompt_len = len(tokenized_prompts.input_ids[i])

        # ...e mascheriamo (con -100) la parte di prompt: la loss deve valutare solo
        # quanto bene il modello genera la RISPOSTA, non quanto bene "prevede" il prompt
        # (che gli viene dato, non generato).
        labels[i, :prompt_len] = label_pad_token_id

    # Mascheriamo anche gli eventuali token di padding nella parte di risposta (righe
    # dove attention_mask == 0): non sono token "reali", quindi non devono contribuire
    # alla loss.
    # Questo sarebbe implicitamente gestito dall'ignore_index di CrossEntropyLoss, ma è
    # più chiaro (e sicuro) farlo esplicitamente qui.
    # Possiamo usare l'attention mask per farlo.
    labels[tokenized_complete.attention_mask == 0] = label_pad_token_id

    return {
        "input_ids": tokenized_complete.input_ids,
        "attention_mask": tokenized_complete.attention_mask,
        "labels": labels,
    }
