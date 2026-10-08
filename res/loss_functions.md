# Le loss di AntiDote spiegate da zero

**Scopo di questo documento**: fornire una mappa completa, da zero, di ogni loss function usata nel progetto — cosa calcola, perché esiste, dove vive nel codice e come si combina con le altre. È il documento "ombrello" da cui partire; per il caso specifico del difetto D4 (termine `safety_clean`) vedi [D4_ancora_sicurezza_clean_zero_to_hero.md](D4_ancora_sicurezza_clean_zero_to_hero.md), che lo approfondisce con dati sperimentali.

**Codice canonico**: `selfsrc/losses.py` (loss), `selfsrc/immunise.py` (dove si combinano), `selfsrc/attack.py` e `selfsrc/evaluate.py` (dove si riusano per attaccare/misurare). Esiste anche una versione storica quasi gemella in `loss.py`/`training.py` (root): la segnalo solo dove rilevante, perché contiene bug noti già corretti in `selfsrc/`.

---

## 0. In una frase

AntiDote addestra un **difensore** (un adapter LoRA su un LLM) a resistere a un **avversario** (una rete che genera una "patch" di pesi malevola) usando **quattro loss combinate**: una di sicurezza sotto attacco, una di sicurezza "a riposo" (opzionale, la correzione D4), e due di utilità (non peggiorare sui compiti normali). L'avversario, dal suo lato, usa una **quinta loss** (la stessa macchina DPO, ma orientata al contrario) per imparare ad attaccare meglio. Dopo il training, una **sesta loss** (la solita cross-entropy) viene riusata come "arma" per simulare un vero attacco di fine-tuning e misurare quanto bene il difensore ha retto.

---

## 1. Prerequisiti

### 1.1 Cos'è una loss
Una **loss** è un numero che misura quanto il modello sbaglia rispetto a un obiettivo. Più basso è il numero, meglio il modello soddisfa quell'obiettivo. Il **gradiente** della loss rispetto ai pesi dice in quale direzione muoverli per farla scendere; la **backpropagation** calcola quel gradiente; l'**optimizer** (qui `AdamW`) applica lo spostamento. Quando un modello ha *più* loss contemporaneamente, ognuna "tira" i pesi in una direzione diversa, e il risultato finale è un compromesso.

### 1.2 Log-probabilità di una risposta
Un LLM autoregressivo assegna una probabilità a ogni token condizionata ai precedenti. La log-probabilità di un'intera risposta $y$ dato un prompt $x$ è:
$$\log P(y \mid x) = \sum_{t=1}^{T} \log P(y_t \mid x, y_{<t})$$
Si usa il *log* perché moltiplicare tante probabilità (numeri piccoli, tra 0 e 1) farebbe collassare il prodotto a zero numericamente; la somma di log è equivalente e stabile.

### 1.3 Shift autoregressivo (dettaglio che ricorre in ogni loss)
In un modello causale, il logit prodotto alla posizione $i$ è la previsione del token alla posizione $i+1$ (il modello non vede ancora quel token). Per confrontare logit e label bisogna quindi "shiftare" di una posizione: si scarta l'ultimo logit (non ha un token successivo da prevedere) e il primo token delle label (nessun logit lo prevede). Lo si trova identico in `log_p_loss`, in `DPOLoss.get_batch_logps` e implicitamente in `compute_kl_loss`.

### 1.4 Masking con `-100`
I dataset mascherano con la label `-100` tutto ciò che non deve contribuire alla loss (il prompt, il padding). `torch.nn.CrossEntropyLoss` ignora automaticamente quell'indice (è il suo `ignore_index` di default); le loss scritte a mano (DPO, KL) devono invece gestirlo esplicitamente con una maschera booleana.

### 1.5 LoRA e modello di riferimento
Il **difensore** non è il modello intero: è un piccolo *adapter* LoRA (poche matrici aggiuntive) applicato a un modello base congelato. Il **modello di riferimento** (`ref_model`) è una copia congelata dello stato "prima del training" (in questo progetto, spesso lo stesso modello con l'adapter *spento*, per non duplicare la memoria — vedi `model_setup.ReferenceModel`). Serve da "punto zero" rispetto al quale si misurano i cambiamenti: diverse loss (DPO, KL) non guardano la probabilità assoluta, ma lo *spostamento* rispetto al riferimento.

### 1.6 Il gioco bi-livello (min-max)
AntiDote risolve, concettualmente:
$$\min_{\theta}\ \max_{\phi}\ \mathcal{L}_{\text{dannosa}}\big(\theta + \Delta_\phi(\theta)\big) \quad \text{soggetto a} \quad \mathcal{L}_{\text{capacità}}(\theta) \le \varepsilon$$
- $\theta$ = pesi del difensore (l'adapter LoRA).
- $\phi$ = pesi dell'avversario (l'hypernetwork).
- $\Delta_\phi(\theta)$ = la patch LoRA malevola generata dall'avversario osservando le attivazioni del difensore.

Si alternano due fasi per ogni "blocco" di training (`selfsrc/immunise.py`, funzione `run_immunisation`):
- **Fase 1 (Adversary)**: $\theta$ congelato, si allena $\phi$ per *massimizzare* la probabilità di risposta dannosa sotto patch.
- **Fase 2 (Defender)**: $\phi$ congelato, si allena $\theta$ per *minimizzare* quella stessa probabilità (cioè restare sicuro) anche sotto la patch, mantenendo l'utilità sui compiti benigni.

Tutte le loss che seguono servono a implementare, un pezzo alla volta, questo schema.

---

## 2. Loss di linguaggio: cross-entropy next-token

**Dove**: `log_p_loss` — [selfsrc/losses.py:31-66](../selfsrc/losses.py) (identica in `loss.py:15-56` nella root) · wrapper `compute_lm_loss` — [selfsrc/losses.py:114-133](../selfsrc/losses.py)

### 2.1 Cosa calcola
È la cross-entropy standard di un modello di linguaggio: quanto il modello è "sorpreso" dal token vero che segue, in media su tutti i token di risposta (non del prompt, grazie alla maschera `-100`).
$$\mathcal{L}_{\text{LM}} = -\frac{1}{N}\sum_{t=1}^{N} \log P_\theta(y_t \mid x, y_{<t})$$
dove $N$ è il numero di token di risposta non mascherati. In codice: shift di logit/label, `view(-1, vocab_size)` per appiattire batch e sequenza, poi `torch.nn.CrossEntropyLoss()`.

Un dettaglio implementativo non banale: `resolve_vocab_size` ([selfsrc/losses.py:97-111](../selfsrc/losses.py)) recupera `vocab_size` da `model.config` invece che da `model.vocab_size`, perché quest'ultimo attributo non è garantito su un `PeftModel` (che inoltra gli attributi al modello sottostante solo in parte) nelle versioni recenti di `transformers`.

### 2.2 Perché esiste (tre ruoli diversi, stessa funzione)
È usata in tre punti del progetto con scopi concettualmente diversi — ma è sempre la stessa identica cross-entropy:

1. **Retain/utilità del difensore** — `immunise.py:413` (`loss_lm`): calcolata su istruzioni *benigne*, senza patch avversaria. Serve a impedire che il difensore, nel diventare sicuro, dimentichi come scrivere testo normale. È il termine "CE" dell'equazione 6 del paper.
2. **Obiettivo dell'attacco di fine-tuning post-hoc** — `attack.py:175` (`objective="sft_harmful"`): *dopo* il training, si simula un vero attaccante che fa fine-tuning diretto sulle risposte dannose. L'attaccante non ha bisogno di DPO o di un riferimento: gli basta la cross-entropy, il fine-tuning più semplice possibile. Qui la stessa loss che "insegna a scrivere bene" viene puntata su contenuto dannoso.
3. **Metrica di valutazione dell'utilità** — `evaluate.py:110-127` (`utility_loss`): proxy della *Fine-tune Accuracy* del paper. Più basso = il modello immunizzato è rimasto più vicino al comportamento originale su testo benigno.

### 2.3 Perché una sola funzione per tre scopi
Non è un caso di riuso casuale: concettualmente la cross-entropy non "sa" se il testo che le viene dato è benigno o dannoso, sicuro o pericoloso — minimizza semplicemente la sorpresa sul prossimo token. È il *dataset* che le viene passato (benigno vs. dannoso) a determinare se il suo effetto è "mantenere l'utilità" o "insegnare contenuto dannoso". È la stessa logica per cui un fine-tuner malevolo reale non ha bisogno di tecniche sofisticate: la cross-entropy, applicata a dati dannosi, è già un attacco efficace.

---

## 3. Loss DPO: il cuore delle fasi Adversary e Defender

**Dove**: classe `DPOLoss` — [selfsrc/losses.py:166-439](../selfsrc/losses.py) · funzione pronta all'uso `compute_dpo_loss` — [selfsrc/losses.py:461-497](../selfsrc/losses.py) · log-prob di riferimento `compute_reference_logps` — [selfsrc/losses.py:442-458](../selfsrc/losses.py)

### 3.1 L'idea del DPO (Direct Preference Optimization)
Riferimento: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290). Dato un prompt $x$ e due risposte — una preferita (`chosen`, $y_w$) e una scartata (`rejected`, $y_l$) — si vuole che il modello preferisca $y_w$ a $y_l$ *più di quanto già facesse il modello di riferimento*. Non serve un modello di reward separato (come nell'RLHF classico): la preferenza si legge direttamente dal rapporto di log-probabilità.

Si definisce:
$$\Delta_{\text{policy}} = \log\pi_\theta(y_w\mid x) - \log\pi_\theta(y_l\mid x), \qquad \Delta_{\text{ref}} = \log\pi_{\text{ref}}(y_w\mid x) - \log\pi_{\text{ref}}(y_l\mid x)$$
$$\text{logits} = \Delta_{\text{policy}} - \Delta_{\text{ref}}$$
`logits` è "quanto più, rispetto al riferimento, il modello in training preferisce ora $y_w$ su $y_l$". È proprio questa quantità che le quattro varianti di DPO cercano di spingere in alto, ciascuna con una forma diversa:

| `loss_type` | Formula | Comportamento | Codice |
|---|---|---|---|
| `sigmoid` (paper originale) | $-\log\sigma(\beta\cdot\text{logits})$ | satura a 0 quando `logits` è già grande; spinge all'infinito altrimenti | [losses.py:396-399](../selfsrc/losses.py) |
| `hinge` | $\max(0,\, 1-\beta\cdot\text{logits})$ | nessuna penalità sopra soglia (stile SVM), lineare sotto | [losses.py:403](../selfsrc/losses.py) |
| **`ipo`** (default del progetto) | $(\text{logits} - \tfrac{1}{2\beta})^2$ | penalizza la *distanza* da un obiettivo fisso, non spinge all'infinito | [losses.py:404-409](../selfsrc/losses.py) |
| `kto_pair` | variante con termine KL per-batch come riferimento | non accoppia chosen/rejected esempio per esempio | [losses.py:410-425](../selfsrc/losses.py) |

Il progetto usa **IPO** di default (`selfsrc/config.json:134`, `beta=0.1`): è più stabile su dataset piccoli e rumorosi, perché non ha la coda "a spingere all'infinito" della variante sigmoid originale — importante con un modello piccolo (Qwen-0.5B nei trial) dove i gradienti possono già essere estremi.

### 3.2 Come si calcolano le log-probabilità (`get_batch_logps`)
[selfsrc/losses.py:207-260](../selfsrc/losses.py). Stesso shift autoregressivo di `log_p_loss`, poi per ogni token si estrae $\log P(\text{token reale}) = \text{logit}_{\text{token}} - \text{logsumexp}(\text{logit})$ (identità della softmax in forma logaritmica, numericamente stabile). Due scelte importanti, entrambe configurabili:

- **Somma vs. media per token** (`average_log_prob`, default `False`): la somma rende le risposte lunghe "pesare" più nella loss; su un modello piccolo con batch rumorosi questo fa esplodere la scala della loss IPO (il progetto nota valori $\sim 10^4$-$10^5$ e gradienti dell'adversary fino a $10^{18}$-$10^{19}$, da cui la necessità di `clip_or_skip`, §5). La media normalizza per lunghezza.
- **Maschera `-100`**: i token mascherati sono temporaneamente sostituiti con l'indice 0 (serve solo perché `torch.gather` non accetta indici negativi), poi azzerati con la maschera booleana — il valore 0 "fittizio" non entra mai nel risultato.

Un dettaglio di efficienza degno di nota: la versione originale (notebook) calcolava `logits.log_softmax(-1)` per *tutto* il vocabolario prima di estrarre il token di interesse — tenendo in memoria una seconda copia (batch × sequenza × ~152k vocaboli) dei logit nel grafo computazionale. La riscrittura in `selfsrc/` usa invece l'identità $\log\text{softmax}(q)_{\text{label}} = q_{\text{label}} - \text{logsumexp}(q)$, risultato matematicamente identico ma senza materializzare quella copia enorme.

### 3.3 `compute_dpo_loss`: una funzione, due (anzi tre) significati
[selfsrc/losses.py:461-497](../selfsrc/losses.py). È la funzione che il loop di training chiama davvero: calcola le log-prob della policy, le confronta con quelle di riferimento (già calcolate a parte da `compute_reference_logps`, §3.4) e ritorna la media delle loss per-esempio. **Non fa il backward**: lo fa chi la chiama, in [selfsrc/immunise.py](../selfsrc/immunise.py).

La stessa identica funzione viene invocata **tre volte per ogni passo del difensore**, cambiando solo quale modello riceve il forward e con quale orientamento chosen/rejected è stato costruito il batch:

- **$\mathcal{L}_{\text{adv}}$** (Fase 1, Adversary) — `immunise.py:314`. Batch con l'orientamento *grezzo* del dataset (`chosen` = risposta dannosa, `rejected` = risposta sicura). Bassa quando, **sotto la patch avversaria**, il modello preferisce la risposta dannosa. L'avversario minimizza questa loss aggiornando solo i propri parametri (l'hypernetwork) — minimizzarla equivale a "massimizzare l'attacco", perché l'orientamento del batch è già quello di un attaccante.
- **$\mathcal{L}_{\text{safety}}$** (Fase 2, Defender) — `immunise.py:398`. Stesso batch, ma con `chosen`/`rejected` **scambiati** ([immunise.py:366-373](../selfsrc/immunise.py)): ora bassa quando, sotto la stessa patch, il modello preferisce la risposta *sicura*. Questo scambio esplicito è la correzione del "FIX #4" rispetto al notebook originale, che nella sua forma più antica aveva l'orientamento invertito per errore.
- **$\mathcal{L}_{\text{clean}}$** (Fase 2, Defender, opzionale — "Knob B"/difetto D4) — `immunise.py:405`. Stesso batch e stesso orientamento di $\mathcal{L}_{\text{safety}}$, ma calcolata **senza** la patch iniettata (il modello "a riposo"). Vedi §6 e il documento dedicato.

### 3.4 Perché il riferimento si calcola separatamente (`compute_reference_logps`)
[selfsrc/losses.py:442-458](../selfsrc/losses.py). Nella modalità "shared" (il riferimento è lo stesso `PeftModel` del difensore, con l'adapter spento, per non duplicare il modello in memoria — vedi `model_setup.ReferenceModel`), il forward del riferimento **deve** avvenire quando la patch avversaria non è iniettata: altrimenti il "punto zero" sarebbe già contaminato dall'attacco. Per questo il chiamante calcola sempre `ref_logps` *prima* di entrare nel blocco `with adversarial_patch(...)` ([immunise.py:296](../selfsrc/immunise.py), [immunise.py:376](../selfsrc/immunise.py)). È anche `@torch.no_grad()`: nessun gradiente deve mai propagarsi attraverso il riferimento.

---

## 4. Loss KL: tenere ferma l'intera distribuzione

**Dove**: `compute_kl_loss` — [selfsrc/losses.py:504-549](../selfsrc/losses.py), usata in `immunise.py:417` (`loss_kl`)

### 4.1 Cosa calcola e perché non basta la LM loss
La LM loss (§2) controlla solo che il *token più probabile* resti ragionevole. Non impedisce che tutta la distribuzione sul vocabolario si deformi (es. il secondo token più probabile cambia completamente), fintanto che il primo resta giusto. La loss KL chiede qualcosa di più forte: che l'**intera distribuzione di probabilità** della policy resti vicina a quella del modello di riferimento, su testo benigno.

Si approssima la vera divergenza KL con una cross-entropy "forward" tra le due distribuzioni:
$$\mathcal{L}_{\text{KL}} \approx -\sum_{v} P_{\text{ref}}(v)\,\log Q_\theta(v)$$
minima quando $P_{\text{ref}} = Q_\theta$. In codice ([losses.py:541](../selfsrc/losses.py)) si usa l'identità $\sum_v P_v \log\text{softmax}(q)_v = \sum_v P_v\, q_v - \text{logsumexp}(q)$ per evitare di materializzare un secondo tensore di `log_softmax` sull'intero vocabolario (stesso trucco di memoria visto in §3.2), poi si media sui soli token non mascherati dall'`attention_mask`.

### 4.2 Perché il riferimento va "congelato" prima
Il modello di riferimento gira sotto `torch.no_grad()` e **prima** del forward della policy ([losses.py:525-528](../selfsrc/losses.py)): in modalità "shared" condivide i pesi col difensore (con l'adapter spento), quindi i suoi logit — grandi, (batch × sequenza × vocabolario) — vanno calcolati e "congelati" (staccati dal grafo) subito, per liberare quella memoria prima di costruire il grafo della policy.

### 4.3 Ruolo nella loss totale
È il secondo termine di "retain" del difensore (insieme a `loss_lm`), corrispondente al termine $\beta\cdot D_{\text{KL}}$ dell'equazione 6 del paper. Peso di default `w_kl = 0.3` ([selfsrc/config.json:144](../selfsrc/config.json)).

---

## 5. Come le loss si combinano nel training: dettagli che contano

### 5.1 La loss totale del difensore
[selfsrc/immunise.py:386-425](../selfsrc/immunise.py):
$$\mathcal{L}_{\text{totale,difensore}} = w_s\cdot\mathcal{L}_{\text{safety}} + w_{sc}\cdot\mathcal{L}_{\text{clean}} + w_{\text{lm}}\cdot\mathcal{L}_{\text{LM}} + w_{\text{kl}}\cdot\mathcal{L}_{\text{KL}}$$
Pesi di default (`selfsrc/config.json:139-145`): $w_s=1.0$, $w_{sc}=0.0$ (disattivato), $w_{\text{lm}}=0.8$, $w_{\text{kl}}=0.3$.

L'avversario invece ha una sola loss, $\mathcal{L}_{\text{adv}}$ (§3.3), senza pesi da combinare.

### 5.2 Backward separati invece di una somma unica
Un dettaglio implementativo che vale la pena capire perché non è ovvio: invece di sommare i quattro termini pesati e fare *un solo* `.backward()` sulla somma, il codice fa il backward di ciascun termine **già moltiplicato per il suo peso**, subito dopo il forward di quel termine ([immunise.py:397-418](../selfsrc/immunise.py)):
```python
with adversarial_patch(model, lora_weights_adv):
    loss_s = compute_dpo_loss(...)
    (w_s * loss_s).backward()
# ... Knob B qui, ancora opzionale ...
loss_lm = compute_lm_loss(...)
(w_lm * loss_lm).backward()
loss_kl = compute_kl_loss(...)
(w_kl * loss_kl).backward()
```
Matematicamente è **identico**: i gradienti si accumulano in `.grad` per linearità della derivata, quindi la somma pesata dei gradienti calcolata in 4 backward separati è uguale al gradiente della somma pesata calcolato in uno. La differenza è di sola memoria: con un backward unico bisognerebbe tenere in vita *contemporaneamente* i grafi computazionali di tutti e quattro i forward (ciascuno grande quanto batch × sequenza × vocabolario); facendo backward uno alla volta, in RAM vive il grafo di un solo forward alla volta. Su un modello che già fatica con la memoria, questo è ciò che permette alla run di non andare in OOM.

### 5.3 Clipping e salto dei passi non finiti
`clip_or_skip` — [selfsrc/immunise.py:211-223](../selfsrc/immunise.py). Dopo il backward, la norma del gradiente viene limitata con `clip_grad_norm_(params, grad_clip)` (default `grad_clip=1.0`). Se quella norma risulta non finita (inf/NaN — capita con la loss IPO su somme di log-prob, dove i gradienti dell'adversary possono arrivare a $10^{18}$-$10^{19}$), il passo viene **saltato** (gradiente azzerato, nessun `optimizer.step()`) invece di essere applicato: senza questa guardia, `clip_grad_norm_` moltiplicherebbe i gradienti per $\text{max\_norm}/\infty = 0$, e $\infty \times 0 = \text{NaN}$ finirebbe direttamente nei pesi, corrompendo il modello in modo irreversibile. I passi saltati sono contati e loggati (`skipped_steps`), non sono invisibili.

### 5.4 Chi vede la patch e chi no
È utile avere chiara la mappa di *quale* loss vede la patch avversaria iniettata e quale no:

| Loss | Patch iniettata? | Dati |
|---|---|---|
| $\mathcal{L}_{\text{adv}}$ | sì | dannosi |
| $\mathcal{L}_{\text{safety}}$ | sì | dannosi |
| $\mathcal{L}_{\text{clean}}$ (opz.) | **no** | dannosi |
| $\mathcal{L}_{\text{LM}}$ | no | benigni |
| $\mathcal{L}_{\text{KL}}$ | no | benigni |

Il buco storico (prima di Knob B) è evidente guardando la tabella per righe: sui dati *dannosi*, esiste una riga "con patch" ma nessuna "senza patch". È esattamente il difetto D4 (§6).

---

## 6. Il Knob B / `safety_clean`: la quinta loss, e perché serve

**Dove**: [selfsrc/immunise.py:401-407](../selfsrc/immunise.py) · approfondimento completo con dati sperimentali: [D4_ancora_sicurezza_clean_zero_to_hero.md](D4_ancora_sicurezza_clean_zero_to_hero.md)

Senza questo termine, il difensore riceve un segnale di sicurezza **solo quando è sotto attacco** (patch iniettata). A riposo, sui prompt dannosi, non c'è nessun vincolo — e il modo più economico per il difensore di resistere alla patch può passare per uno spostamento dei pesi che, preso da solo (senza patch), peggiora il comportamento. `safety_clean` chiude questo buco: è la **stessa** loss DPO di sicurezza, sullo stesso batch e con gli stessi `ref_logps` già calcolati per $\mathcal{L}_{\text{safety}}$, ma eseguita *fuori* dal blocco `with adversarial_patch(...)` — quindi sul modello pulito. Con peso $w_{sc}=0$ (default) il comportamento è identico al progetto originale: è pensata come un'ablazione a costo zero, un solo parametro da cambiare.

Per la storia completa — perché il problema nasce, come si misura (`d_safe`), i risultati sperimentali e i loro limiti statistici — vedi il documento dedicato linkato sopra: non la ripeto qui per evitare che i due documenti divergano nel tempo.

---

## 7. Dopo il training: le stesse loss, usate per attaccare e misurare

AntiDote non si ferma all'immunizzazione: `selfsrc/attack.py` ed `selfsrc/evaluate.py` **riusano** le loss già viste per rispondere alla domanda "ha funzionato?".

### 7.1 L'attacco di valutazione (resistenza al tampering)
[selfsrc/attack.py](../selfsrc/attack.py), funzione `run_harmful_finetune_attack` (righe 90-189). Non è una nuova loss: è `compute_lm_loss` (§2) usata come un vero fine-tuning SFT diretto sulle risposte dannose — `objective="sft_harmful"` ([attack.py:115-121](../selfsrc/attack.py)) è l'unico supportato. Con `harmful_ratio < 1.0` ([attack.py:125-126](../selfsrc/attack.py)) si mescolano batch dannosi e benigni (il paper usa un mix 20:80), per simulare un attacco più "realistico" (un vero attaccante di solito non ha solo dati dannosi).

### 7.2 Le metriche derivate (non loss, ma costruite sulle stesse quantità)
[selfsrc/evaluate.py](../selfsrc/evaluate.py):
- **Safety margin**: $\text{sm}_i = \log P(y_{\text{sicura}}\mid x_i) - \log P(y_{\text{dannosa}}\mid x_i)$ per ogni prompt dannoso — calcolato con lo stesso `DPOLoss.concatenated_forward` usato in training ([evaluate.py:48-60](../selfsrc/evaluate.py)). Media positiva = il modello preferisce, in media, la risposta sicura.
- **TRR (Tamper Resistance Recovered)**: frazione del danno da attacco che l'immunizzazione recupera,
$$\text{TRR} = \frac{\text{sm}_{\text{immunizzato, sotto attacco}} - \text{sm}_{\text{base, sotto attacco}}}{\text{sm}_{\text{base, pulito}} - \text{sm}_{\text{base, sotto attacco}}}$$
- **$d_{\text{safe}}$**: degrado a riposo, $\text{sm}_{\text{immunizzato, pulito}} - \text{sm}_{\text{base, pulito}}$ — la metrica che motiva Knob B (§6).
- **`utility_loss`**: la cross-entropy media su istruzioni benigne (§2.2, punto 3) — proxy della capacità residua del modello.

Un dettaglio di coerenza sperimentale: `make_eval_dpo_util` ([evaluate.py:130-143](../selfsrc/evaluate.py)) crea una `DPOLoss` **dedicata alla valutazione**, con `beta` e `loss_type` fissi (non presi dal config del trial): se ogni trial misurasse il margine con impostazioni diverse, due run non sarebbero confrontabili.

---

## 8. Codice morto: loss scritte ma non usate

Due funzioni, presenti solo nella root `loss.py`, non sono chiamate da nessun loop di training (né in `training.py`, né in `selfsrc/`):

- **`log_1_minus_p_loss`** — [loss.py:59-148](../loss.py). Calcola $\log(1 - P(\text{label}))$ tramite un logsumexp mascherato che esclude il logit del token corretto. L'idea sarebbe un obiettivo di **unlearning**: minimizzare (non massimizzare) la probabilità del token corretto, con una soglia (`threshold=-15.0`) per non sprecare gradiente su token su cui il modello è già "abbastanza sconfitto". Non collegata a nessuna fase del progetto.
- **`max_entropy_loss`** — [loss.py:151-175](../loss.py). Entropia negativa media: minimizzarla equivale a *massimizzare* l'entropia della distribuzione, cioè spingere il modello verso previsioni più incerte/uniformi (utile, in teoria, per "confondere" deliberatamente il modello su certi input). Anche questa, mai richiamata.

Non sono bug: sono probabilmente residui di esplorazioni precedenti o idee alternative non portate a termine. Vale la pena saperle leggere (potrebbero spiegare scelte di design altrove) ma non descrivono alcun comportamento della pipeline attuale.

---

## 9. Tabella riassuntiva

| # | Loss | Formula (sintetica) | File:riga | Chiamata da | Peso default | Ruolo |
|---|---|---|---|---|---|---|
| 1 | Cross-entropy / LM (`log_p_loss`, `compute_lm_loss`) | $-\text{mean}_t \log P(y_t\mid y_{<t})$ | [losses.py:31-66,114-133](../selfsrc/losses.py) | `immunise.py:413`, `attack.py:175`, `evaluate.py:125` | $w_{\text{lm}}=0.8$ | Utilità (defender) · attacco SFT · metrica utility |
| 2 | DPO (IPO) — adversary | $(\text{logits}-\tfrac{1}{2\beta})^2$, orientamento grezzo | [losses.py:166-497](../selfsrc/losses.py) | `immunise.py:314` | — | Fase 1: imparare ad attaccare |
| 3 | DPO (IPO) — safety | stessa classe, chosen/rejected scambiati | stessa | `immunise.py:398` | $w_s=1.0$ | Fase 2: resistere sotto patch |
| 4 | DPO (IPO) — safety_clean (Knob B / D4) | stessa classe, senza patch | stessa | `immunise.py:405` | $w_{sc}=0.0$ | Ancora di sicurezza a riposo |
| 5 | KL (cross-entropy forward) | $-\sum_v P_{\text{ref}}(v)\log Q_\theta(v)$ | [losses.py:504-549](../selfsrc/losses.py) | `immunise.py:417` | $w_{\text{kl}}=0.3$ | Retain: intera distribuzione vicina al riferimento |
| 6 | `log_1_minus_p_loss` | $\log(1-P(\text{label}))$ con soglia | [loss.py:59-148](../loss.py) | nessuno | — | Codice morto (unlearning) |
| 7 | `max_entropy_loss` | $-\text{entropia media}$ | [loss.py:151-175](../loss.py) | nessuno | — | Codice morto (confusione) |

---

## 10. Glossario rapido

| Termine | Significato |
|---|---|
| **Policy** | Il modello attualmente in training (il difensore, con o senza patch secondo la fase) |
| **Reference / `ref_model`** | Copia congelata del modello, punto zero per DPO e KL |
| **`chosen` / `rejected`** | Le due risposte di una coppia DPO; quale sia "preferita" dipende dall'orientamento scelto dal chiamante |
| **`logits` (nel contesto DPO)** | Non i logit del modello: è $\Delta_{\text{policy}} - \Delta_{\text{ref}}$, il "vantaggio" relativo di chosen su rejected |
| **β (beta)** | Temperatura della DPO: più basso = il modello può allontanarsi di più dal riferimento |
| **Patch avversaria** | Matrici (U, V) generate dall'hypernetwork e iniettate temporaneamente nel difensore durante il forward |
| **Retain** | Insieme di loss (LM + KL) che preservano il comportamento benigno originale |
| **Safety margin (sm)** | $\log P(\text{sicura}) - \log P(\text{dannosa})$ su un prompt dannoso |
| **TRR** | Frazione del danno da attacco recuperata dall'immunizzazione |
| **$d_{\text{safe}}$** | Degrado del margine di sicurezza a riposo causato dal training (vedi D4) |

---

## 11. Da qui, dove andare

- Per il caso di studio completo sul termine `safety_clean` (dati sperimentali, limiti statistici, soluzioni proposte): [D4_ancora_sicurezza_clean_zero_to_hero.md](D4_ancora_sicurezza_clean_zero_to_hero.md).
- Per la derivazione formale di ciascuna equazione del paper: `res/equation1_tutorial.md` … `equation6_tutorial.md`.
- Per la mappa dei moduli e l'elenco dei bug corretti rispetto al notebook originale: [selfsrc/README.md](../selfsrc/README.md).
