import torch
import torch.nn as nn
from collections import defaultdict

# Questa classe serve a "spiare" (senza modificarli) gli input o gli output di certi
# sotto-moduli di una rete neurale, mentre questa fa un forward pass normale.
# Il meccanismo che PyTorch offre per farlo si chiama "forward hook": è una funzione
# che si registra su un modulo, e che PyTorch chiama automaticamente ogni volta che
# quel modulo finisce il suo forward pass, passandole input e output.
# In AntiDote, questo serve all'ADVERSARY: per generare una patch LoRA "su misura" per
# il prompt corrente, ha bisogno di vedere le attivazioni interne (cioè i vettori
# numerici che scorrono dentro il modello) nei layer che vuole attaccare.


class ActivationCache:
    def __init__(self, model: nn.Module, target_modules: list, reshape: bool = True, capture_output: bool = False):
        # Il modello su cui vogliamo "spiare" le attivazioni.
        self.model = model
        # Nomi "brevi" dei sotto-moduli da osservare (es. "q_proj", "k_proj", ...): ogni
        # modulo del modello il cui nome finale corrisponde a uno di questi verrà
        # osservato (vedi `register_hooks`).
        self.target_modules = target_modules
        # Se True, appiattiamo il tensore catturato da (batch, seq_len, dim) a
        # (batch*seq_len, dim): più comodo da passare in seguito a reti che si aspettano
        # un input 2D (vedi come viene usato in adversary.py).
        self.reshape = reshape
        # Se True, catturiamo l'OUTPUT del modulo; se False (default), catturiamo il suo
        # INPUT. In AntiDote vogliamo l'input di q/k/v/o_proj (cioè cosa "entra" in quella
        # proiezione), quindi il default è False.
        self.capture_output = capture_output
        # Dizionario {nome_modulo: tensore_catturato}. All'inizio è vuoto: si popola
        # quando registriamo gli hook (con valore None) e si riempie ad ogni forward pass.
        self.activations = {}
        # Lista degli "handle" degli hook registrati: servono per poterli rimuovere in
        # seguito con `remove_hooks()`.
        self.hooks = []
        # Dispositivo (CPU/GPU) su cui si trova il modello: usato per spostare i tensori
        # catturati sullo stesso dispositivo, per evitare errori "tensori su device diversi".
        self.device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device('cpu')

    def _get_hook(self, module_name: str, is_output: bool = False):
        """
        Costruisce (e ritorna) la funzione hook effettiva da registrare su un modulo.
        Usiamo una "closure" (funzione dentro funzione) così ogni hook "ricorda" a quale
        nome di modulo appartiene, senza doverlo passare ogni volta.
        """
        def hook(module, input, output):
            # `input` è una tupla di argomenti posizionali passati al modulo: per un
            # `nn.Linear` (come q_proj/k_proj/...) il primo (e unico) argomento è il
            # tensore in ingresso, quindi prendiamo `input[0]`. Se invece vogliamo
            # l'output, PyTorch ce lo passa già come tensore singolo.
            tensor = output if is_output else input[0]
            if self.reshape:
                # Appiattiamo tutte le dimensioni tranne l'ultima (quella delle feature):
                # da (batch, seq_len, dim) a (batch*seq_len, dim).
                tensor = tensor.reshape(-1, tensor.shape[-1])
            # `.detach()`: stacchiamo il tensore dal grafo computazionale di autograd,
            # perché questa è solo una "fotografia" da leggere, non deve propagare
            # gradiente all'indietro attraverso questa cache.
            self.activations[module_name] = tensor.detach().to(self.device)  # Keep on same device for efficiency
        return hook

    def register_hooks(self):
        """
        Cerca dentro il modello tutti i sotto-moduli il cui nome (l'ultimo pezzo, dopo
        l'ultimo punto) corrisponde a uno di `target_modules`, e registra un hook su
        ciascuno di essi.

        ATTENZIONE: chiamare questo metodo più volte SENZA prima chiamare
        `remove_hooks()` registra hook aggiuntivi sopra quelli già presenti, invece di
        sostituirli: per questo in training.py lo si chiama una sola volta, prima del
        loop di training, invece di richiamarlo ad ogni step (vedi il commento "FIX #2"
        in training.py).
        """
        registered_count = 0

        # `model.named_modules()` restituisce (nome_completo, modulo) per OGNI
        # sotto-modulo della rete, a qualsiasi profondità (es.
        # "base_model.model.layers.3.self_attn.q_proj").
        for name, module in self.model.named_modules():
            # Prendiamo solo l'ultimo pezzo del nome (dopo l'ultimo punto), che è il nome
            # "locale" del modulo (es. "q_proj").
            module_parts = name.split('.')
            module_name = module_parts[-1]

            # Se questo modulo è uno di quelli che ci interessano...
            if module_name in self.target_modules:
                hook_fn = self._get_hook(name, self.capture_output)
                # `register_forward_hook` ritorna un "handle": lo salviamo per poter
                # rimuovere l'hook più avanti.
                self.hooks.append(module.register_forward_hook(hook_fn))
                # Pre-registriamo la chiave nel dizionario (con valore None), così
                # esiste già anche prima del primo forward pass.
                self.activations[name] = None
                registered_count += 1

    def clear_cache(self):
        """Svuota solo i VALORI delle attivazioni salvate, mantenendo le chiavi (i nomi
        dei moduli) già note. Va chiamato prima di ogni nuovo forward pass di cui si
        vogliono catturare attivazioni "fresche", per non confondere i valori di step
        diversi."""
        for key in self.activations:
            self.activations[key] = None

    def remove_hooks(self):
        """Rimuove tutti gli hook registrati (chiamando `.remove()` su ciascun handle).
        Va sempre chiamato quando non servono più, altrimenti gli hook restano attaccati
        al modello per sempre, continuando a "spiare" (e a consumare tempo/memoria) ogni
        futuro forward pass, anche fuori da qualsiasi ciclo di training."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        # Permette di usare questa classe come context manager: `with act_cache: ...`
        # registra gli hook all'ingresso del blocco `with`.
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ATTENZIONE (bug noto, lasciato volutamente com'era): questo metodo dovrebbe
        # rimuovere gli hook registrati da `__enter__` (chiamando `self.remove_hooks()`),
        # ma la riga sottostante è commentata, quindi in pratica NON fa nulla ("no-op").
        # Il risultato è che, se si usa `with act_cache:` più volte (ad es. dentro un
        # ciclo di training), ad ogni giro si aggiungono NUOVI hook sopra quelli già
        # presenti, senza mai rimuoverli: gli hook si accumulano senza limite.
        # Per questo motivo, in training.py NON si usa più il pattern `with act_cache:`
        # dentro al ciclo: gli hook si registrano una volta sola con `register_hooks()`
        # prima del ciclo, si puliscono i valori con `clear_cache()` ad ogni step, e si
        # rimuovono con `remove_hooks()` (esplicito) solo alla fine.
        # self.remove_hooks()
        pass
