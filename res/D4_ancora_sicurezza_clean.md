# Difetto D4 spiegato da zero: la mancanza di un'ancora di sicurezza sul modello pulito

**Riferimenti**: `selfsrc/hpo/paced_report_ita.md` §7.4 (difetto D4), §10 (lezione 2), §4 (metriche).
**Codice**: `selfsrc/immunise.py`, righe 363–425 (fase di attacco del difensore).
**Scopo di questo documento**: capire D4 abbastanza a fondo da spiegarlo in tesi, valutarne i limiti e proporre soluzioni.

> **Nota sul collegamento con "10.2"**: nel report non esiste una sottosezione 10.2. La sezione 10 ha i punti 1–4. Il collegamento corretto è il **punto 2 di §10** ("I modelli piccoli non hanno inerzia sufficiente da soli… l'ancora di sicurezza sul pulito ($\mathcal{L}_{\text{clean\_safe}}$) è assolutamente indispensabile"). È la lezione generale che D4 illustra nel caso specifico di Qwen-0.5B.

---

## 0. In una frase

Il difensore viene addestrato a rifiutare richieste dannose **solo mentre un avversario gli inietta una perturbazione**. Nessun termine della loss gli chiede di rifiutare anche quando il modello è pulito. Di conseguenza, per resistere alla perturbazione, il difensore può deformare i pesi in modo da peggiorare il comportamento a riposo. D4 è questa deformazione. La correzione, chiamata **Knob B** (`safety_clean`), aggiunge una seconda loss di sicurezza calcolata sul modello pulito.

---

## 1. Prerequisiti da zero

### 1.1 Modello linguistico e log-probabilità
Un LLM assegna una probabilità a ogni sequenza di token. Per confrontare due risposte $y_a$ e $y_b$ alla stessa domanda $x$, si usano le log-probabilità:
$$\log P(y \mid x) = \sum_{t} \log P(y_t \mid x, y_{<t})$$
Il report usa la versione **normalizzata per token**, cioè divisa per la lunghezza della risposta, così che risposte lunghe non siano penalizzate solo per la lunghezza.

### 1.2 Margine di sicurezza
Dato un prompt dannoso $x_i$, con risposta sicura $y_{\text{safe}}$ e dannosa $y_{\text{harmful}}$:
$$\text{sm}_i = \log P(y_{\text{safe}} \mid x_i) - \log P(y_{\text{harmful}} \mid x_i)$$
- $\text{sm}_i > 0$: il modello preferisce la risposta sicura.
- $\text{sm}_i < 0$: il modello preferisce quella dannosa.
- Il **valore medio** su $N = 128$ prompt è la metrica `safety_margin` (sm).

### 1.3 Fine-tuning e LoRA
- **Fine-tuning**: si continua l'addestramento di un modello già pronto su nuovi dati.
- **LoRA**: invece di modificare tutti i pesi, si aggiungono piccole matrici addestrabili (un *adapter*). Nel progetto il **difensore è un adapter LoRA**; il modello base resta congelato (riga 348–350 di `immunise.py`).

### 1.4 Loss e backpropagation
Una **loss** è un numero che misura quanto il modello sbaglia. Il **gradiente** indica in quale direzione spostare i pesi per diminuirla. La **backpropagation** calcola questi gradienti. Ogni termine della loss viene ridotto dall'ottimizzatore, quindi termini diversi "tirano" i pesi in direzioni diverse.

### 1.5 DPO (Direct Preference Optimization)
La DPO insegna al modello a preferire una risposta `chosen` rispetto a una `rejected`, senza un modello di ricompensa separato. Ha due ingredienti:
$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(\beta\,\big[\underbrace{\Delta_{\text{policy}}}_{\log\pi_\theta(y_w)-\log\pi_\theta(y_l)} - \underbrace{\Delta_{\text{ref}}}_{\log\pi_{\text{ref}}(y_w)-\log\pi_{\text{ref}}(y_l)}\big]\Big)$$
- $y_w$ = chosen, $y_l$ = rejected, $\pi_\theta$ = modello in addestramento, $\pi_{\text{ref}}$ = modello di riferimento congelato, $\beta$ = temperatura.
- **Punto cruciale**: la DPO è **relativa** al riferimento. Non chiede al modello di avere probabilità alta per $y_w$ in assoluto. Chiede che il *vantaggio* di $y_w$ su $y_l$ cresca rispetto a quello che il riferimento già aveva.
- Quando il margine supera abbastanza il riferimento, $\sigma(\cdot)$ si avvicina a 1 e il gradiente si annulla. La loss **satura**: smette di spingere.

Nel difensore, lo scambio `chosen ↔ rejected` nel codice (righe 366–373) fa sì che la **risposta sicura** sia quella marcata come `chosen`. In questo modo la DPO spinge il modello verso la risposta sicura.

### 1.6 Due tipi di "retain" (conservare il comportamento)
- **LM loss** su testo benigno: il modello deve continuare a predire bene il testo normale (utilità).
- **KL** su testo benigno: la distribuzione completa sul vocabolario non deve allontanarsi dal riferimento (`compute_kl_loss` in `selfsrc/losses.py`, riga 504).

**Attenzione**: entrambe lavorano su dati **benigni**. Questo è il punto da cui nasce D4.

### 1.7 Min-max (gioco bi-livello)
AntiDote risolve
$$\min_{\theta} \max_{\phi} \mathcal{L}_{\text{dannosa}}(\theta + \Delta_{\phi}(\theta)) + \mathcal{L}_{\text{utilità}}(\theta)$$
- **Avversario** ($\phi$): una piccola rete che genera una perturbazione $\Delta_\phi$ dei pesi. Vuole massimizzare la probabilità di risposte dannose.
- **Difensore** ($\theta$, l'adapter LoRA): vuole minimizzare la stessa loss, cioè restare sicuro *anche con la perturbazione addosso*.
- In pratica: a ogni blocco, l'avversario si allena (fase 1), poi il difensore si allena contro la perturbazione attuale (fase 2).

---

## 2. Come funziona la fase del difensore in AntiDote

In `immunise.py`, nella fase 2 (righe 352–425), per ogni passo il difensore riceve **quattro termini di loss**:

| Termine | Dati | Modello | Scopo | Riga |
|---|---|---|---|---|
| $\mathcal{L}_{\text{safety}}$ (`loss_s`) | prompt dannosi | **con patch avversaria** | resistere all'attacco | 398–399 |
| $\mathcal{L}_{\text{clean}}$ (`loss_sc`) | prompt dannosi | **senza patch** | ancora sul modello pulito (**Knob B**) | 405–406 |
| $\mathcal{L}_{\text{LM}}$ (`loss_lm`) | testo benigno | senza patch | utilità linguistica | 413–414 |
| $\mathcal{L}_{\text{KL}}$ (`loss_kl`) | testo benigno | senza patch, vs riferimento | non allontanarsi dal riferimento | 417–418 |

Totale:
$$\mathcal{L}_{\text{totale}} = w_{s}\,\mathcal{L}_{\text{safety}} + w_{sc}\,\mathcal{L}_{\text{clean}} + w_{\text{lm}}\,\mathcal{L}_{\text{LM}} + w_{\text{kl}}\,\mathcal{L}_{\text{KL}}$$

> Nota di coerenza: il report scrive $w_{\text{adv}}$, mentre nel codice il peso è `weights["safety"]`. È lo stesso termine. Conviene usare un solo nome nella tesi.

Il riquadro da notare è quello **dei dati dannosi senza patch**. Prima di Knob B è vuoto.

---

## 3. Il difetto D4 in dettaglio

### 3.1 Il problema
Nel design originale, il difensore riceve un segnale di sicurezza **solo sotto attacco**. Guardando la tabella sopra, con $w_{sc} = 0$ la riga "dannosi senza patch" non esiste. Quindi:

- **Sotto attacco**: "rifiuta i prompt dannosi" (termine $\mathcal{L}_{\text{safety}}$).
- **A riposo, prompt dannosi**: nessun termine.
- **A riposo, prompt benigni**: LM e KL, che proteggono solo l'utilità.

Il comportamento a riposo sui prompt dannosi è quindi **non vincolato**. Il report lo chiama "mancanza di un'ancora".

### 3.2 Perché questo fa danni: l'intuizione
Immagina il difensore come un punto $\theta$ in uno spazio di pesi. L'avversario applica uno spostamento $\Delta$, e il difensore deve trovare un $\theta$ che funzioni bene per **tutti** gli spostamenti che l'avversario può generare (è una famiglia di perturbazioni, non una sola).

Il gradiente spinge $\theta$ verso una regione dove la sicurezza regge contro quella famiglia. Non c'è però nessun vincolo che dica: "e nel punto $\theta$ stesso (senza perturbazione) la sicurezza deve restare buona". Il punto non perturbato è un caso particolare della famiglia, ma non è garantito che sia buono: il modo più economico di resistere alla perturbazione può passare per un $\theta$ che, preso da solo, ha un margine peggiore.

Riassunto del meccanismo, come lo descrive il report:
> "il modello distorceva i propri pesi, rovinando il suo margine di sicurezza sul modello pulito"

Questa è un'**ipotesi meccanistica** coerente con i dati, non una dimostrazione diretta. Per dimostrarla servirebbe, ad esempio, misurare la norma di $\theta - \theta_0$ o la direzione del cambiamento.

### 3.3 Perché la KL non basta
Un'obiezione naturale: "c'è già la KL, non protegge il modello?". La KL protegge la distribuzione sui **testi benigni**. Sui prompt dannosi, il modello può spostarsi liberamente. Il degrado di sicurezza passa quindi nel punto cieco tra utilità (protetta) e sicurezza a riposo (non protetta).

### 3.4 La metrica che misura il danno: $d_{\text{safe}}$
$$d_{\text{safe}} = \text{sm}_{\text{immunizzato, pulito}} - \text{sm}_{\text{base, pulito}}$$
- È il **degrado a riposo** causato dall'immunizzazione.
- Obiettivo del progetto: $d_{\text{safe}} \ge -0.02$.
- Nei trial senza Knob B il report osserva $d_{\text{safe}} = -0.227$. È il sintomo di D4.

---

## 4. La correzione: Knob B (`safety_clean`)

Nel codice (righe 401–407):
```python
# Knob B (D4): ancora la sicurezza pulita (senza patch adversariale)
loss_sc_val = 0.0
if w_sc > 0.0:
    monitor.check_memory()
    loss_sc = compute_dpo_loss(model, defender_dpo_batch, ref_logps, dpo_util)
    (w_sc * loss_sc).backward()
    loss_sc_val = loss_sc.item()
```

Punti importanti:
1. Usa **lo stesso batch** `defender_dpo_batch` e **gli stessi** `ref_logps` della loss sotto attacco. Cambia solo la presenza della patch (la chiamata è fuori dal blocco `with adversarial_patch(...)`).
2. I `ref_logps` sono calcolati **prima** della patch (riga 376), quindi il riferimento è il modello originale, non quello attaccato.
3. Con $w_{sc} = 0$ il comportamento è identico al design originale. Questo permette un'**ablazione pulita**: stesso codice, un solo parametro cambia.

Forma matematica:
$$\mathcal{L}_{\text{clean}} = -\log\sigma\!\Big(\beta\,\big[\Delta_{\text{policy}}(\theta) - \Delta_{\text{ref}}\big]\Big) \quad \text{(senza patch)}$$

**Intuizione**: ora il difensore deve soddisfare due richieste insieme. "Rifiuta anche con la perturbazione" e "rifiuta anche senza". Il secondo vincolo lo tiene vicino al proprio stato sicuro originale.

---

## 5. Le evidenze del report, lette criticamente

| Trial | $w_{sc}$ | TRR | $\pm$ rumore 2σ | $d_{\text{safe}}$ | Commento |
|---|---|---|---|---|---|
| t23 | 0.0 | +0.170 | ±0.23 | −0.227 | sintomo di D4 |
| t24 | 0.0 | +0.132 | ±0.23 | n.d. | reset dell'avversario |
| t25 | 1.0 | +0.235 | ±0.233 | −0.158 | primo TRR sopra il rumore |
| t26 | 2.0 | +0.279 | ±0.233 | −0.118 | TRR e $d_{\text{safe}}$ migliorano insieme |
| t30 | 2.0 | +0.447 | ±0.232 | −0.032 | run lungo (8 blocchi), combinato |

### Cosa è solido
- **$d_{\text{safe}}$ migliora in modo monotono** con $w_{sc}$: −0.227 → −0.158 → −0.118. Il report stima l'errore standard appaiato di $d_{\text{safe}}$ a circa $0.033$. La differenza t23 → t26 (0.109) è circa 3 errori standard. È l'evidenza più forte a favore di D4.
- Il **meccanismo** coerente: aggiungere un termine sul modello pulito tiene sotto controllo il degrado a riposo.

### Cosa è debole (da dire chiaramente in tesi)
- **Il TRR non distingue i trial.** Le differenze t23 → t25 → t26 sono $+0.065$ e $+0.044$, dentro il rumore di $\pm 0.23$. Il TRR di t25 ($0.235$) supera la soglia di $0.233$ di un soffio.
- **Un solo seme per trial.** Il report stesso mostra in §9 che un seme diverso cambia molto il risultato (t31: TRR $+0.046$, $\text{sm}_{\text{base, att}} = -0.738$ contro $-0.443$). Le differenze tra t23–t26 potrebbero essere in parte varianza di seme.
- **Confondenti in t30.** Il "risultato straordinario" combina più cambiamenti (gradient clipping D2, pool D5, $w_{sc}$ D4, 8 blocchi invece di 4). Non isola l'effetto di D4. Non si può attribuire a D4 il salto da $+0.279$ a $+0.447$.
- **$d_{\text{safe}} = -0.032$ in t30** è ancora sotto la soglia $-0.02$ del progetto, sia pure entro circa un errore standard. Il report lo presenta come "AZZERATO": è una formulazione più forte di quanto i dati permettano.
- **Soglia di successo**: il report usa "TRR > 0.232 = fuori dal rumore" per ogni singolo trial. Il rumore di ogni trial è stimato su 128 esempi, quindi è una soglia per singola misura, non per il confronto tra trial.

Conclusione onesta per la tesi: **il dato più solido è $d_{\text{safe}}$, non il TRR**. Il TRR è la metrica "regina", ma è quella con più rumore.

---

## 6. Il legame con la lezione 2 di §10

Il punto 2 della sezione 10 dice:
> I modelli giganti (7B o 70B) hanno pesi pre-addestrati così stabili che mantengono la sicurezza pulita anche con poca regolarizzazione. Nei modelli da 0.5B, l'allenamento avversariale sbilancia rapidamente le rappresentazioni: l'ancora di sicurezza sul pulito è indispensabile.

Questa è la **generalizzazione** di D4:
- D4 è il fenomeno: senza ancora, $d_{\text{safe}}$ peggiora.
- La lezione 2 è l'interpretazione: il fenomeno dipende dalla **capacità** del modello. Su 0.5B, la perturbazione avversaria sposta molto le rappresentazioni, quindi la regolarizzazione implicita non basta.

**Attenzione**: la lezione 2 è un'**ipotesi di scala**, basata su un solo modello (Qwen-0.5B). Non è stata testata su modelli più grandi nel report. In tesi va presentata come ipotesi, non come legge.

---

## 7. Come spiegare D4 in una frase per ciascun pubblico

- **Per chi non è esperto**: "Il sistema imparava a difendersi dagli attacchi, ma nel farlo dimenticava di comportarsi bene quando non c'era attacco. Abbiamo aggiunto un controllo che gli ricorda il comportamento normale."
- **Per un ML engineer**: "La loss di sicurezza era calcolata solo sotto perturbazione avversaria, senza vincolo sul punto non perturbato. Il degrado a riposo ($d_{\text{safe}}$) cresceva di conseguenza. Il termine `safety_clean` (DPO sul modello senza patch, stesso batch e stessi reference logps) ha ridotto $d_{\text{safe}}$ da $-0.227$ a $-0.118$ con $w_{sc}=2$."
- **Per la commissione**: "È un caso di degrado della sicurezza a riposo indotto dall'addestramento avversariale. Lo abbiamo diagnosticato con un'ablazione controllata e mitigato con un regolarizzatore. L'evidenza è forte sul degrado, più debole sul guadagno di robustezza, e richiede verifica multi-seme."

---

## 8. Limiti e domande aperte

1. **Robustezza statistica**: servono più semi per trial (almeno 3–5) per separare l'effetto di $w_{sc}$ dal rumore di seme.
2. **Isolamento dell'effetto**: un'ablazione con tutto fisso tranne $w_{sc} \in \{0, 0.5, 1, 2\}$ e run lunghi, non brevi.
3. **Scala del modello**: la lezione 2 è testata su un solo modello. Un confronto con un modello più grande (anche solo 1.5B) confermerebbe o smentirebbe l'ipotesi.
4. **Meccanismo**: l'ipotesi "i pesi si spostano in una direzione che degrada il punto pulito" non è misurata direttamente. Si potrebbe misurare $\|\theta - \theta_0\|$ e la sicurezza lungo il segmento tra $\theta_0$ e $\theta$.
5. **Scelta di $w_{sc}$**: il report prova solo $0$, $1$, $2$. Non si sa se $w_{sc} > 2$ peggiori l'utilità o aiuti ancora.
6. **Saturazione DPO**: la loss DPO satura quando il margine supera il riferimento. Con $w_{sc}$ alto, il termine clean potrebbe smettere di contribuire. Vale la pena controllare la norma del gradiente di $\mathcal{L}_{\text{clean}}$ nei log.

---

## 9. Possibili soluzioni (ipotesi da verificare, non risultati)

Queste sono proposte per la tesi. **Nessuna è stata testata nel progetto.**

1. **Estendere la KL ai prompt dannosi.** Oggi la KL vale solo su testo benigno (§3.3). Applicare la KL anche ai prompt dannosi sul modello pulito tratterebbe il problema alla radice, non con un termine di sicurezza aggiuntivo. Rischio: può irrigidire il modello sul comportamento di riferimento, riducendo la capacità di imparare.
2. **Vincolo sul modello pulito con margine minimo** invece di una DPO pura. Una hinge loss del tipo $\max(0, \delta - \text{sm}_{\text{clean}})$ non satura come la DPO e impone un margine minimo esplicito $\delta$.
3. **Early stopping su $d_{\text{safe}}$.** Il config ha già `abort_if_d_safe_below = -0.22`. Renderlo più stretto (es. $-0.05$) è una modifica a costo zero e già supportata dal codice.
4. **Selezione del checkpoint** con un obiettivo congiunto (es. TRR soggetto a $d_{\text{safe}} \ge -0.02$), invece del checkpoint finale.
5. **Riferimento diverso per la KL**: usare il riferimento pulito come ancora per tutti i termini, non solo per quello benigno.

Per la tesi, la soluzione 1 è la più principled e la più facile da giustificare, ma richiede validazione multi-seme prima di essere presentata come risultato.

---

## 10. Glossario rapido

| Termine | Significato |
|---|---|
| **sm** / `safety_margin` | Media di $\log P(\text{safe}) - \log P(\text{harmful})$ su N prompt dannosi |
| **safe_pref_rate** | Percentuale di prompt in cui $\text{sm}_i > 0$ |
| **TRR** | Frazione del danno da attacco recuperata: $\frac{\text{sm}_{\text{imm,att}} - \text{sm}_{\text{base,att}}}{\text{sm}_{\text{base,pulito}} - \text{sm}_{\text{base,att}}}$ |
| **$d_{\text{safe}}$** | Degrado a riposo dell'immunizzato rispetto al base pulito |
| **Rumore 2σ** | Banda di incertezza al 95% sul TRR, circa $\pm 0.23$ nel report |
| **Patch avversaria** | Perturbazione dei pesi generata dall'avversario e iniettata durante il forward |
| **DPO** | Loss che spinge la risposta `chosen` sopra la `rejected` rispetto a un riferimento |
| **Ancora / Knob B / `safety_clean`** | Termine DPO calcolato sul modello pulito con peso $w_{sc}$ |
| **KL retain** | Penalità che tiene la distribuzione su testo benigno vicina al riferimento |
| **Ablazione** | Esperimento che cambia una sola variabile per isolarne l'effetto |

---

## 11. Riferimenti nel codice

- Fase difensore e loss: [selfsrc/immunise.py](selfsrc/immunise.py) righe 352–425
- Patch avversaria: `adversarial_patch(...)` riga 397 (definizione in `selfsrc/injection.py`)
- Loss DPO: `compute_dpo_loss` in [selfsrc/losses.py](selfsrc/losses.py) riga 461
- Loss KL: `compute_kl_loss` in [selfsrc/losses.py](selfsrc/losses.py) riga 504
- Loss LM: `compute_lm_loss` in [selfsrc/losses.py](selfsrc/losses.py) riga 114
- Report sorgente: [selfsrc/hpo/paced_report_ita.md](selfsrc/hpo/paced_report_ita.md) §7.4, §9, §10
