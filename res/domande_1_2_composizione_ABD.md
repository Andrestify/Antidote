# Domande 1 e 2: come si combinano A, B, D durante il training e nel modello finale

**Collegato a**: [D4_ancora_sicurezza_clean.md](D4_ancora_sicurezza_clean.md)
**Codice citato**: `selfsrc/immunise.py`, `selfsrc/attack.py`
**Scopo**: chiarire, da zero, cosa significa "combinare" base, attacco e difesa in due momenti diversi — durante l'ottimizzazione del difensore (Domanda 1) e nel modello finale rilasciato (Domanda 2) — e perché sono due domande diverse, non la stessa domanda ripetuta.

---

## 0. Notazione usata in questo documento

| Simbolo | Significato |
|---|---|
| **B** | Il modello Transformer di base (pre-addestrato / instruct), **congelato** — i suoi pesi non cambiano mai durante l'immunizzazione. |
| **A** | L'adapter/patch LoRA generato dall'**avversario** (l'iperrete $\phi$): una perturbazione temporanea dei pesi, pensata per simulare un attacco. |
| **D** | L'adapter LoRA del **difensore** $\theta$: l'unica parte che viene effettivamente allenata durante l'immunizzazione. |
| **A\*, D\*** | Le versioni **allenate** (pesi finali) di A e D, dopo che il training è terminato. |
| **"X+Y"** | Significa: "il modello B con sopra applicato l'adapter/patch Y (ed eventualmente anche X)", **non** una somma algebrica generica — vedi §3 per il perché questo conta. |

---

## 1. Prerequisiti da zero (ripasso veloce)

Se hai già letto il file collegato su D4, questa sezione è solo un ripasso; altrimenti è il minimo indispensabile per seguire le risposte.

### 1.1 LoRA / adapter
Invece di modificare tutti i pesi di un modello enorme, si aggiungono piccole matrici addestrabili ("adapter") che producono una correzione $\Delta W$ a specifiche matrici di peso. Il modello risultante è $W_{\text{usato}} = W_B + \Delta W$. **Un adapter LoRA non è un modello a sé stante**: è definito come una correzione *di* una matrice di base. Senza quella matrice di base, l'adapter non ha nulla a cui applicarsi — torneremo su questo in §3.

### 1.2 Batch
Un gruppo di esempi (prompt+risposta) processato insieme per calcolare una loss e i relativi gradienti in un colpo solo, invece che uno a uno.

### 1.3 Reference logps
Le log-probabilità che il modello **di riferimento** (il checkpoint congelato, prima di qualunque modifica di D in questo step) assegna alle risposte del batch. Servono come "punto fisso" di paragone nella loss DPO: la DPO non chiede "rendi la risposta sicura probabile in assoluto", chiede "aumenta la sua preferenza *rispetto a quanto il riferimento già la preferiva*".

### 1.4 `w_sc` (peso di Knob B / `safety_clean`)
Il coefficiente che moltiplica il termine di loss "sicurezza sul modello pulito" (senza patch A). Se `w_sc = 0`, quel termine non esiste (è il difetto D4, descritto nel file collegato); se `w_sc > 0`, il difensore viene spinto anche a restare sicuro senza la patch addosso.

### 1.5 `.backward()`, `.grad`, AdamW
- **`.backward()`**: calcola il gradiente (quanto e in che direzione spostare ogni peso per diminuire la loss), tramite backpropagation. Non modifica i pesi.
- **`.grad`**: l'attributo dove PyTorch **accumula** il gradiente calcolato. Se chiami `.backward()` più volte senza azzerare `.grad` in mezzo, i contributi **si somman**o.
- **AdamW**: l'algoritmo che usa `.grad` per aggiornare davvero i pesi, con medie mobili adattive per parametro più una penalità di weight decay.

### 1.6 SFT (Supervised Fine-Tuning)
Addestramento standard: dato prompt → risposta desiderata, si minimizza la loss di next-token-prediction solo sulla risposta. È il tipo di attacco usato in valutazione (§4).

### 1.7 DPO (richiamo)
$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(\beta\big[\Delta_{\text{policy}} - \Delta_{\text{ref}}\big]\Big)$$
dove $\Delta = \log P(\text{chosen}) - \log P(\text{rejected})$. Nel difensore, dopo lo scambio chosen↔rejected, "chosen" è la risposta sicura.

---

## 2. Domanda 1 — durante il training, quale configurazione si usa per ottimizzare D?

### La domanda
> Durante l'ottimizzazione del difensore, per calcolare la loss di sicurezza usiamo la configurazione "attacco + base + difesa" (A+B+D), oppure la configurazione "solo base + difesa" (B+D, senza l'attacco)?

### Risposta
**Nessuna delle due in esclusiva: si usano entrambe, come due termini di loss separati nello stesso step.**

Nel codice (`selfsrc/immunise.py`, righe 397–407), per ogni passo della fase difensore ci sono due forward pass distinti sugli *stessi* dati dannosi:

```python
with adversarial_patch(model, lora_weights_adv):        # forward: A+B+D
    loss_s = compute_dpo_loss(model, defender_dpo_batch, ref_logps, dpo_util)
    (w_s * loss_s).backward()

if w_sc > 0.0:                                            # forward: B+D (niente patch)
    loss_sc = compute_dpo_loss(model, defender_dpo_batch, ref_logps, dpo_util)
    (w_sc * loss_sc).backward()
```

| Termine | Configurazione | Dati | Chi riceve il gradiente |
|---|---|---|---|
| $\mathcal{L}_{\text{safety}}$ | **A+B+D** (patch iniettata) | prompt dannosi | solo D (A e B sono `requires_grad=False`) |
| $\mathcal{L}_{\text{clean}}$ (Knob B) | **B+D** (nessuna patch) | stessi prompt dannosi | solo D |

Entrambi i `.backward()` scrivono su `.grad` degli stessi parametri (solo l'adapter D ha `requires_grad=True`: righe 349–350, `p.requires_grad = adapter_name in n`). I due contributi **si accumulano** prima che l'ottimizzatore faccia un singolo `step()`.

### Perché conta
- **Prima di Knob B** ($w_{sc}=0$): il difensore veniva aggiornato sui prompt dannosi **solo** tramite A+B+D. Nessun segnale gli diceva "resta sicuro anche senza A". È la radice del difetto D4.
- **Con Knob B**: D riceve nello stesso step sia "resisti con A addosso" sia "resta sicuro anche senza A" — da cui il miglioramento di $d_{\text{safe}}$ osservato nel report.

### Perché "A+D" (senza B) non ha senso e non esiste nel codice

La formulazione originale della domanda proponeva come seconda opzione "A+D". Questa combinazione **non è praticabile**, per due motivi, uno concettuale e uno implementativo:

1. **Motivo concettuale**: un adapter LoRA (sia A sia D) non è definito come un modello a sé stante, ma come una **correzione** $\Delta W$ a matrici di peso che *appartengono a B*. La formula stessa del gioco min-max lo rende esplicito:
   $$\min_{\theta} \max_{\phi} \mathcal{L}\big(\theta + \Delta_{\phi}(\theta)\big)$$
   Qui $\Delta_\phi(\theta)$ (cioè A) è una perturbazione **applicata a** $\theta$ (i pesi, che includono B). Non esiste un modo di scrivere "solo A+D" perché A è definita *in funzione di* B — rimuovere B lascia A senza nulla su cui agire, come dire "la correzione di un numero, senza il numero".
2. **Motivo implementativo**: nel codice, `make_adversarial_patch` genera le perturbazioni a partire dalla cache delle attivazioni di B (`act_cache`, calcolata facendo un forward **attraverso B**). La funzione `adversarial_patch(model, lora_weights_adv)` (in `selfsrc/injection.py`) inietta quella perturbazione **dentro l'oggetto `model`**, che è già B con D applicato come adapter. Non c'è nessun punto del codice in cui A viene combinata con D *senza* che B sia presente nel grafo computazionale — semplicemente non è un'operazione che il codice permette di eseguire, perché tecnicamente A e D sono entrambe correzioni iniettate sulle stesse matrici di B, non oggetti indipendenti che si possano somministrare "a vuoto".

In sintesi: **A+D senza B non è "sconsigliato", è privo di significato** — è come chiedere il risultato di una correzione senza il valore da correggere.

---

## 3. Domanda 2 — di cosa è fatto il modello finale immunizzato?

### La domanda
> Il modello finale "immunizzato" (ottenuto dopo il training, con il difensore allenato D\*) è composto da "attacco allenato + base + difesa allenata" (A\*+B+D\*), oppure semplicemente da "base + difesa allenata" (B+D\*, senza l'iperrete avversaria)?

### Risposta
**Il modello finale è solo B+D\*. A\* non ne fa parte.**

L'iperrete avversaria A è uno strumento **esclusivamente di allenamento**: serve a generare, durante il training, perturbazioni economiche contro cui D si allena a resistere (è l'oggetto della Domanda 1). Finito il training, **A\* viene scartata**: non è salvata nel checkpoint consegnato, non è distribuita, non è richiesta per usare il modello.

Il prodotto che rilasci/usi è: modello base B (congelato, invariato) + adapter LoRA D\* (allenato) — punto.

### Ma allora cos'è "immunizzato, attaccato" nelle metriche?

Qui serve un chiarimento importante, perché è facile confonderlo con la Domanda 1. Ho verificato il codice (`selfsrc/attack.py`, funzione `run_harmful_finetune_attack`, righe 90–173): la condizione **"attaccato" usata in fase di valutazione non usa A\* in nessun modo**. È un **vero attacco SFT** (fine-tuning supervisionato standard, con un proprio `AdamW`) che continua ad allenare **direttamente i parametri di D\*** su un piccolo dataset di prompt+risposte dannose:

```python
for p in model.parameters():
    p.requires_grad = False
for p in defender_params:
    p.requires_grad = True
...
optimizer = AdamW(defender_params, lr=lr)
...
pairs = {"prompt": raw["prompt"], "response": raw["chosen"]}  # risposta DANNOSA
```

Questo è esattamente il tipo di attacco descritto nella sezione 2 del report (Harmful Fine-Tuning): un vero attaccante che scarica il modello e lo fine-tuna su esempi dannosi. Non è la rete A\* che genera una patch — è gradient descent reale che sposta $D^*$ stesso in un nuovo punto $D^*_{\text{attaccato}}$.

$$\text{"immunizzato, attaccato"} = B + D^*_{\text{attaccato}}$$

### Tabella riassuntiva dei tre momenti

| Momento | Configurazione | A\* coinvolta? |
|---|---|---|
| **Training** (ottimizzare D, Domanda 1) | A+B+D *(termine safety)* **e** B+D *(termine clean)* | sì, solo nel primo termine, come proxy di attacco |
| **Modello finale rilasciato** (Domanda 2) | B + D\* | no, mai |
| **Valutazione "attaccato"** (Domanda 2, caso particolare) | B + D\*ₐₜₜ (D\* ulteriormente fine-tunato con un vero attacco SFT) | no — attacco reale, meccanismo diverso da A |

### Perché questa distinzione è metodologicamente importante
Se si valutasse la robustezza del modello finale riapplicando A\* (la stessa rete con cui si è allenato D), si rischierebbe di misurare solo "D ha imparato a battere A\*" — un risultato circolare, non una vera garanzia di sicurezza. Usare invece un attacco SFT reale e indipendente (diverso meccanismo, nessuna condivisione di pesi con l'addestramento) è quello che rende la metrica TRR una misura credibile di robustezza, e non solo di "overfitting all'avversario di allenamento".

---

## 4. Dubbi che potrebbero sorgere (FAQ)

**D: Se D riceve gradiente sia da A+B+D sia da B+D nello stesso step, i due termini non potrebbero "tirare" in direzioni opposte e annullarsi?**
R: Possono essere parzialmente in conflitto (è esattamente il trade-off che il peso $w_{sc}$ regola). Non si annullano matematicamente — si *sommano* come vettori, quindi il risultato è una direzione di compromesso. Il report osserva empiricamente che aumentare $w_{sc}$ migliora $d_{\text{safe}}$ senza far collassare il TRR, ma non è garantito in generale: un $w_{sc}$ troppo alto potrebbe far prevalere il termine clean a scapito della robustezza sotto attacco (non testato esplicitamente nel report con $w_{sc} > 2$).

**D: Perché l'attacco di valutazione aggiorna `defender_params` e non un nuovo adapter separato?**
R: Perché deve simulare esattamente lo scenario reale: un attaccante che scarica il modello pubblicato (B+D\*) e lo fine-tuna. Il modello pubblicato ha solo i parametri di D\* come allenabili/salvati separatamente da B (B è il modello pre-addestrato originale, enorme e condiviso); è quindi naturale, e realistico, che l'attacco agisca proprio su quei parametri.

**D: A\* non viene salvata da nessuna parte, nemmeno per analisi successive?**
R: Dal punto di vista del *prodotto* (il modello da distribuire) no. Nulla vieta di salvare A\* a scopo di analisi/debug durante la ricerca (per esempio per studiare che tipo di perturbazioni genera), ma non è parte del modello immunizzato che un utente finale riceverebbe o dovrebbe installare.

**D: Se A è definita come correzione di B, perché posso comunque "sommare" A+B+D e non, per esempio, A+D+D (due volte D)?**
R: Perché A, B, D non sono termini intercambiabili di una somma algebrica: sono tre oggetti con ruoli diversi — B è la base (una sola, fissa), D è una correzione di B (una sola, quella allenata), A è un'altra correzione di B (temporanea, generata dall'avversario). "A+B+D" è una scorciatoia notazionale per dire "il modello B, con sopra applicate contemporaneamente le correzioni A e D" — non un'operazione aritmetica generica dove i simboli si possono permutare liberamente.

**D: Il fatto che l'attacco di valutazione riparta da D\* e non da zero non lo rende "più facile" per l'attaccante (ha già un buon punto di partenza)?**
R: È corretto, ed è voluto: la domanda scientifica non è "un attaccante che parte da zero riesce a rendere il modello dannoso" (ovvio: un modello non allineato non ha difese). La domanda è "un attaccante che parte dal modello immunizzato e lo fine-tuna con lo stesso identico attacco subito dal modello base, quanto danno riesce a fare in confronto"? Da qui nasce il TRR.

---

## 5. Glossario rapido

| Termine | Significato |
|---|---|
| **B** | Modello Transformer di base, pre-addestrato, congelato |
| **A / A\*** | Adapter LoRA dell'avversario (non allenato / allenato); correzione temporanea di B generata per simulare un attacco |
| **D / D\*** | Adapter LoRA del difensore (non allenato / allenato); l'unica parte effettivamente addestrata nell'immunizzazione |
| **`defender_params`** | I parametri di D, salvati separatamente da B; sono anche il bersaglio del vero attacco in fase di valutazione |
| **Patch avversaria** | L'operazione di iniettare temporaneamente A nel modello durante il forward (context manager `adversarial_patch`) |
| **Knob B / `safety_clean` / `w_sc`** | Il peso del termine di loss che ancora D alla sicurezza anche senza la patch A (correzione di D4) |
| **SFT** | Fine-tuning supervisionato standard; meccanismo del vero attacco in valutazione |
| **TRR** | Frazione del danno da attacco recuperata dal modello immunizzato rispetto al modello base |

---

## 6. Riferimenti nel codice

- Fase difensore, doppio forward A+B+D / B+D: [selfsrc/immunise.py](../selfsrc/immunise.py) righe 349–350, 397–407
- Generazione della patch avversaria da attivazioni di B: `make_adversarial_patch`, [selfsrc/immunise.py](../selfsrc/immunise.py) riga 179
- Iniezione della patch nel modello: `adversarial_patch`, [selfsrc/injection.py](../selfsrc/injection.py) riga 179
- Attacco reale di valutazione (SFT su `defender_params`): `run_harmful_finetune_attack`, [selfsrc/attack.py](../selfsrc/attack.py) righe 90–173
- Documento collegato sul difetto D4: [D4_ancora_sicurezza_clean.md](D4_ancora_sicurezza_clean.md)
- Report sorgente: [selfsrc/hpo/paced_report_ita.md](../selfsrc/hpo/paced_report_ita.md)
