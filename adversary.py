import torch
import torch.nn as nn
import torch.nn.functional as F

# L'ADVERSARY è una piccola rete neurale ("hypernetwork": una rete che genera i PESI di
# un'altra rete, invece di generare direttamente un output finale) che impara a produrre
# patch LoRA malevole su misura, guardando le attivazioni interne del modello per un dato
# prompt. In AntiDote, l'adversary simula un attaccante che prende il modello e cerca di
# "ri-addestrarlo" per farlo comportare in modo dannoso: il DEFENDER (in training.py) si
# allena a restare sicuro anche quando questa patch viene iniettata.


class Adversary(nn.Module):
    def __init__(self, r, layer_configs: dict, enc_dim = 1024, num_heads = 16):
        """
        Inizializza un adversary generico, capace di generare patch per layer
        eterogenei (con forme diverse, es. q_proj vs k_proj in modelli con
        Grouped-Query Attention).

        :param r: rango (r) delle matrici LoRA generate (più piccolo = patch più
            "semplice"/vincolata, più grande = patch più espressiva ma con più
            parametri).
        :param layer_configs: dizionario che mappa un nome di "tipo di layer" (es.
            "q_proj") alla sua forma (in_features, out_features).
        :param enc_dim: dimensione interna (nascosta) usata dalla parte "condivisa"
            della rete (l'aggregatore di attenzione + il corpo FFN sotto).
        :param num_heads: numero di teste per il layer di multi-head attention interno.
        """
        super().__init__()
        self.r = r
        self.layer_configs = layer_configs

        # --- 1. Teste di proiezione in INGRESSO, specializzate per dimensione ---
        # Layer diversi (q_proj, k_proj, ...) possono avere `in_features` diversi tra
        # loro (soprattutto con Grouped-Query Attention). Per poter comunque usare la
        # stessa rete "condivisa" sotto, prima proiettiamo ogni tipo di input alla
        # STESSA dimensione fissa `enc_dim`, usando una testa lineare diversa per ogni
        # dimensione di ingresso distinta osservata in `layer_configs`.
        self.input_projs = nn.ModuleDict()
        unique_in_dims = set(cfg[0] for cfg in layer_configs.values())
        for in_dim in unique_in_dims:
            self.input_projs[str(in_dim)] = nn.Linear(in_dim, enc_dim)

        # --- 2. Il "corpo" CONDIVISO della rete (uguale per tutti i tipi di layer) ---
        # Un layer di self-attention (le attivazioni "guardano" se stesse per capire
        # quali posizioni della sequenza sono più rilevanti)...
        self.aggregator = nn.MultiheadAttention(embed_dim=enc_dim, num_heads=num_heads)
        # ...seguito da due blocchi feed-forward residuali (vedi classe ResidualFFN
        # sotto), che raffinano ulteriormente la rappresentazione.
        self.body = nn.Sequential(
            ResidualFFN(enc_dim),
            ResidualFFN(enc_dim),
        )

        # --- 3. Teste di proiezione in USCITA, specializzate per tipo di layer ---
        # Ogni tipo di layer bersaglio ha bisogno di una coppia di matrici LoRA (U, V)
        # con forme diverse: U (LoRA "A") ha forma (r, in_features), V (LoRA "B") ha
        # forma (out_features, r). Per ognuno, quindi, serve una testa di uscita diversa
        # che generi il numero giusto di valori.
        self.U_heads = nn.ModuleDict()
        self.V_heads = nn.ModuleDict()

        for config_name, (in_dim, out_dim) in layer_configs.items():
            # La testa U produce r*in_features numeri (che verranno poi rimodellati
            # nella matrice U di forma (r, in_features)).
            self.U_heads[config_name] = nn.Linear(enc_dim, r * in_dim)
            # La testa V produce out_features*r numeri (rimodellati in (out_features, r)).
            self.V_heads[config_name] = nn.Linear(enc_dim, out_dim * r)

        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializza i pesi di tutti i layer lineari con Xavier/Glorot normale (una
        strategia di inizializzazione comune, pensata per mantenere una varianza simile
        tra input e output all'inizio del training) e i bias a zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, activations: torch.Tensor, config_name: str):
        """
        Genera la coppia di matrici LoRA (U, V) per il layer indicato da `config_name`,
        a partire dalle attivazioni catturate su quel layer (vedi ActivationCache).

        Args:
            activations: tensore delle attivazioni catturate, forma (B, L, in_dim) dove
                B è la dimensione di batch, L la lunghezza di sequenza, in_dim la
                dimensione delle feature di quel layer.
            config_name: nome del tipo di layer (es. "q_proj"), usato per scegliere le
                teste di ingresso/uscita corrette.
        """
        in_dim, out_dim = self.layer_configs[config_name]
        # activations arriva con forma (B, L, V) dove V = in_dim per questo layer.

        # 1. Selezioniamo la testa di INGRESSO corretta in base alla dimensione delle
        # feature (in_dim), e proiettiamo le attivazioni alla dimensione condivisa.
        input_proj = self.input_projs[str(in_dim)]
        x = input_proj(activations) # (B, L, enc_dim)

        # 2. Passiamo il risultato attraverso il "corpo" condiviso: prima
        # self-attention (la sequenza "guarda se stessa" per pesare le posizioni più
        # rilevanti), poi i blocchi feed-forward residuali.
        x_attended, _ = self.aggregator(x, x, x)  # (B, L, enc_dim)
        # Facciamo la media lungo la dimensione 0: questo "riassume" la sequenza in
        # un'unica rappresentazione (nota: qui la dimensione 0 è quella di batch/
        # sequenza secondo il formato atteso da nn.MultiheadAttention senza
        # batch_first=True -- il risultato pooled ha comunque la forma attesa dalle
        # teste di uscita sotto).
        x_pooled = x_attended.mean(dim=0, keepdim=True)  # (1, L, enc_dim)

        # 3. Selezioniamo le teste di USCITA corrette per questo tipo di layer.
        U_head = self.U_heads[config_name]
        V_head = self.V_heads[config_name]

        U_flat = U_head(x_pooled)
        V_flat = V_head(x_pooled)

        # 4. Rimodelliamo i vettori "piatti" prodotti dalle teste nelle vere matrici
        # LoRA, con le forme che si aspetta il resto del codice (vedi peft_injection.py):
        # U ha forma (L, r, in_dim), V ha forma (L, out_dim, r). Qui L (che nella riga
        # sopra viene trattato come la dimensione di "batch" per la moltiplicazione a
        # blocchi in peft_injection.py) corrisponde al numero di "righe" prodotte da
        # x_pooled dopo il pooling.
        U = U_flat.view(-1, self.r, in_dim)       # Shape: (L, r, in_dim)
        V = V_flat.view(-1, out_dim, self.r)      # Shape: (L, out_dim, r)

        return U, V


class ResidualFFN(nn.Module):
    """
    Un blocco feed-forward "alla Transformer" con connessione residua: normalizza
    l'input, lo passa attraverso due layer lineari con una non-linearità (GELU) in
    mezzo, e infine SOMMA il risultato all'input originale (invece di sostituirlo).
    La connessione residua (`x + o3`) rende più facile l'addestramento di reti
    profonde, perché il gradiente ha sempre un "percorso diretto" all'indietro anche
    se il blocco impara inizialmente qualcosa di poco utile.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # Espandiamo la dimensione di un fattore 3 (un "collo di bottiglia inverso",
        # comune nei blocchi feed-forward dei Transformer) prima di tornare a `dim`.
        self.up_proj = nn.Linear(dim, dim * 3)
        self.gelu = nn.GELU()
        # LayerNorm: normalizza ogni vettore (media 0, varianza 1) prima di elaborarlo,
        # per stabilizzare il training.
        self.ln = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim * 3, dim)
        # Dropout: durante il training, spegne casualmente una frazione delle unità,
        # per rendere la rete più robusta ed evitare che "memorizzi" troppo i dati
        # visti (overfitting).
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Normalizziamo prima di tutto.
        x_norm = self.ln(x)
        # Trasformazione feed-forward: espandi -> non-linearità -> comprimi.
        o1 = self.up_proj(x_norm)
        o2 = self.gelu(o1)
        o3 = self.down_proj(o2)
        o3 = self.dropout(o3)
        # Connessione residua: aggiungiamo il risultato all'input ORIGINALE (non
        # normalizzato), così l'informazione originale non si perde mai del tutto.
        return x + o3
