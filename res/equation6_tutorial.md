# Zero to Hero: Understanding Equation 6 of the AntiDote Paper

*A from-scratch walkthrough of how AntiDote keeps the model useful while it's being hardened against attacks.*

---

## 0. The destination

Equation 6 of the AntiDote paper (Sanyal, Ray & Mandal, 2025, arXiv:2509.08000), from Section 2.3 "The Bi-level Optimization Game," Phase 2 ("The Defender's Turn"), objective 2 ("Capability Objective: Preserving Utility"), is written as:

$$
\mathcal{L}_{\text{cap}}(\theta_D) = \mathbb{E}_{x_c,y_c \sim \mathcal{D}_{\text{cap}}}\big[\mathcal{L}_{\text{CE}}(M_{\theta_{\text{base}}+\theta_D})\big] \;+\; \beta \cdot D_{\text{KL}}\big(P(y\mid x_c;\theta_{\text{base}}+\theta_D) \,\|\, P(y\mid x_c;\theta_{\text{base}})\big) \qquad (6)
$$

computed on the **clean** model $M_{\theta_{\text{base}}+\theta_D}$ — no adversarial patch involved anywhere in this equation.

By the end of this tutorial you'll be able to read this as: *"train the defender to still do ordinary instruction-following well, and simultaneously keep its output behavior close to what the original, unmodified base model would have done — measured with no attack applied at all."* This is the objective that stops AntiDote's safety training from turning the model into a paranoid, unhelpful refuser.

This tutorial assumes you've read [`equation5_tutorial.md`](equation5_tutorial.md) (for $\theta_D$, $\theta_{\text{base}}$, and the general Phase 2 setup). It does **not** assume you already know cross-entropy loss or KL divergence — both are built from scratch below.

---

## 1. Why does this equation exist? The story first

Equation 5 alone has an obvious failure mode. If the *only* thing being trained for is "never prefer harmful outputs," the easiest way to satisfy that is to make the model refuse or hedge on *everything* — harmful and benign requests alike. A model that answers "I can't help with that" to every single prompt would score perfectly on $\mathcal{L}_{\text{safe}}$, and be completely useless.

Equation 6 is the counterweight. It's evaluated **on a completely different data distribution** ($\mathcal{D}_{\text{cap}}$ — ordinary instruction-following and reasoning tasks, not safety probes) and **on a completely different model state** (the *clean* model, with no adversarial patch applied at all — contrast this with Equation 5, which is always evaluated on the attacked $\theta_{\text{adv}}$). This separation is what the paper calls **"gradient purity"** (footnote 2 in the paper, quoted in full in [Section 3.5](#35-why-clean-and-why-that-matters-gradient-purity)): the gradients that shape the model's general capability are never contaminated by the presence of an adversarial patch, so they cleanly reflect only "be a good, helpful model" — nothing about safety or attacks leaks into this signal.

---

## 2. Notation bootcamp

| Symbol | Read it as | Meaning |
|---|---|---|
| $\theta_D$ | "theta-D" | The defender's LoRA weights — the variable Equation 6 is optimized with respect to (same trainable parameter as Equation 5; both objectives update the *same* $\theta_D$, just via different loss terms — see [Section 5](#5-how-equation-6-combines-with-equation-5)). |
| $\theta_{\text{base}}$ | "theta-base" | The frozen, original base model's weights. |
| $M_{\theta_{\text{base}}+\theta_D}$ | "the model at base plus D" | The **clean** defended model: base weights plus the defender's LoRA weights, with **no** hypernetwork attack patch applied. This is the key difference from every equation since Equation 3 — there is no $\theta_{\text{adv}}$ anywhere in Equation 6. |
| $\mathcal{D}_{\text{cap}}$ | "D-cap" | The capability distribution — general-purpose instruction-following and reasoning data (first introduced back in Equation 2's setup), completely separate from the safety-probing data $\mathcal{D}_{\text{safe}}$. |
| $(x_c, y_c) \sim \mathcal{D}_{\text{cap}}$ | "sampled from D-cap" | Draw one example from the capability dataset: a prompt $x_c$ (e.g. a coding question, a math problem, a general instruction) and its single desired high-quality response $y_c$. Note this is a *pair*, not a triplet like $\mathcal{D}_{\text{safe}}$ — there's no "rejected" response here, just one target. |
| $\mathcal{L}_{\text{CE}}(\cdot)$ | "L-C-E", "cross-entropy loss" | The standard language-modeling loss: how well the model predicts the correct next token, averaged across a target response. Built from scratch in [Section 3.2](#32-cross-entropy-teaching-the-model-to-say-the-right-thing). |
| $P(y\mid x_c;\theta)$ | "P of y given x-c, under theta" | The model's probability distribution over possible responses $y$, given prompt $x_c$, under parameters $\theta$ — the same conditional-probability object from [`equation1_tutorial.md`](equation1_tutorial.md), just evaluated at two different parameter settings here. |
| $D_{\text{KL}}(P \,\|\, Q)$ | "KL divergence of P from Q" | A measure of how different one probability distribution $P$ is from a reference distribution $Q$. Built from scratch in [Section 3.3](#33-kl-divergence-measuring-drift-from-the-original-model). |
| $\beta$ | "beta" | A hyperparameter weighting how strongly the KL term is enforced. The paper sets $\beta = 0.3$ (see [Section 5](#5-how-equation-6-combines-with-equation-5)). |

---

## 3. Assembling the pieces: term by term

### 3.1 The overall shape: two terms, two different jobs

$$\mathcal{L}_{\text{cap}}(\theta_D) = \underbrace{\mathbb{E}\big[\mathcal{L}_{\text{CE}}(M_{\theta_{\text{base}}+\theta_D})\big]}_{\text{"be accurate on capability data"}} \;+\; \beta\cdot\underbrace{D_{\text{KL}}(\cdots)}_{\text{"don't drift from the original model"}}$$

These two terms encode two related but distinct ideas about "preserving utility": the first says *do well on held-out instruction/reasoning examples*; the second says *stay close, in general output behavior, to the model you started from*. Either term alone would be an incomplete notion of "capability preservation" — together they're complementary.

### 3.2 Cross-entropy: teaching the model to say the right thing

Start from the most basic training signal in all of language modeling. Given a prompt $x_c$ and a desired response $y_c$ (a sequence of tokens $y_c = (t_1, t_2, \ldots, t_n)$), the model assigns a probability to each token in $y_c$, one at a time, conditioned on everything before it:

$$P(y_c \mid x_c;\theta) = \prod_{i=1}^{n} P(t_i \mid x_c, t_1,\ldots,t_{i-1}; \theta)$$

**Cross-entropy loss** is (up to averaging) just the negative log of this — exactly the same "take the log-probability, negate it" idea used throughout this series of tutorials (compare to [`equation1_tutorial.md`, Section 4.2](equation1_tutorial.md#42-why-a-logarithm)), just now applied as something to *minimize directly*, rather than folded into a preference-comparison margin:

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{n} \log P(t_i \mid x_c, t_1,\ldots,t_{i-1};\theta)$$

Minimizing this pushes the model to assign *higher* probability to each correct next token in the target response $y_c$ — this is exactly the ordinary "next-token prediction" objective used to pre-train and instruction-tune language models in the first place. Here it's being reused to make sure that, as $\theta_D$ gets trained to resist attacks (via Equation 5), it doesn't simultaneously forget how to be a good instruction-follower on ordinary, benign prompts.

### 3.3 KL divergence: measuring drift from the original model

Cross-entropy alone only checks "does the model produce the *one specific* target response well." It says nothing about what the model does for every *other* possible response — two models could both nail the target response $y_c$ on the training data while behaving very differently everywhere else. The KL-divergence term addresses that by directly comparing two full probability distributions over responses.

For two probability distributions $P$ and $Q$ over the same set of outcomes, the KL divergence is:

$$D_{\text{KL}}(P \| Q) = \sum_{y} P(y) \log\frac{P(y)}{Q(y)}$$

A few properties worth internalizing, since KL divergence shows up constantly in ML and is easy to misread:

- $D_{\text{KL}}(P\|Q) \ge 0$ always, with equality **if and only if** $P$ and $Q$ are exactly the same distribution.
- It is **not symmetric**: $D_{\text{KL}}(P\|Q) \ne D_{\text{KL}}(Q\|P)$ in general — order matters, which is why the paper's notation carefully lists which distribution is "first."
- Intuitively, it measures the extra "surprise" you'd incur, on average, if you believed outcomes came from $Q$ but they actually come from $P$.

In Equation 6, specifically:

$$D_{\text{KL}}\big(P(y\mid x_c;\theta_{\text{base}}+\theta_D) \,\|\, P(y\mid x_c;\theta_{\text{base}})\big)$$

- $P(y\mid x_c;\theta_{\text{base}}+\theta_D)$ — the **current defended model's** distribution over responses to $x_c$ (this is the "$P$" being evaluated — the one that gets to change as $\theta_D$ trains).
- $P(y\mid x_c;\theta_{\text{base}})$ — the **original, untouched base model's** distribution over responses to the same prompt (this is the fixed reference "$Q$").

Minimizing this term pulls the defended model's overall response distribution back toward what the original base model would have done, for the same benign prompt — a direct, explicit anchor against catastrophic forgetting, on top of (not instead of) the cross-entropy term.

### 3.4 The expectation

$$\mathbb{E}_{x_c,y_c\sim\mathcal{D}_{\text{cap}}}\big[\cdots\big]$$

Same idea as every other expectation in this series: sample many $(x_c, y_c)$ pairs from the capability dataset, compute the bracketed quantity for each, and average — turning a per-example score into one trainable scalar.

### 3.5 Why "clean," and why that matters: gradient purity

The paper's own footnote spells this out precisely:

> "By 'gradient purity', we mean that the gradients for the capability objective are computed on the clean model, ensuring they are not 'contaminated' by the presence of the adversarial patch and solely reflect the goal of utility preservation."

Concretely: Equation 6 never touches $\theta_{\text{adv}}$, never calls the hypernetwork $\mathcal{H}_\phi$, and never applies any LoRA attack patch. The forward pass used to compute $\mathcal{L}_{\text{cap}}$ runs purely on $M_{\theta_{\text{base}}+\theta_D}$. Why does this matter?

If, instead, the capability loss *were* computed on the attacked model $\theta_{\text{adv}}$ (the way the safety loss in Equation 5 is), then every gradient flowing back into $\theta_D$ from this term would be entangled with the specific, somewhat arbitrary attack pattern the hypernetwork happened to produce that step. The model might then learn to "preserve capability" in a way that's really just compensating for *that particular patch*, rather than learning a stable, general notion of "stay close to the original model's behavior." By computing it on the clean model instead, the capability gradient is a **stationary target** — it doesn't move around depending on what the adversary is currently doing, which the paper argues is what makes this decoupled approach stable and effective (this is explicitly called out as one of the paper's four headline contributions in the introduction: "Principled Mitigation of the Safety-Utility Trade-off").

---

## 4. A fully worked numerical toy example

Take a benign capability example: $x_c = $ *"What is 15% of 240?"*, $y_c = $ *"36."*

**Cross-entropy term.** Suppose the target response $y_c$ tokenizes to just two tokens, `"36"` and `"."`. Suppose the defended model assigns:

$$P(\texttt{"36"} \mid x_c) = 0.72, \qquad P(\texttt{"."} \mid x_c, \texttt{"36"}) = 0.95$$

$$\mathcal{L}_{\text{CE}} = -\log(0.72) - \log(0.95) \approx 0.329 + 0.051 = 0.380$$

A low cross-entropy value here means the model is doing a good job assigning high probability to the correct tokens; a poorly-preserved model might assign, say, $P(\texttt{"36"}\mid x_c) = 0.10$, giving $\mathcal{L}_{\text{CE}} \approx 2.35$ — a much larger loss, signaling a real regression in basic capability.

**KL term.** Now compare full output distributions over, say, just the first token, simplified to three possible first tokens for illustration (`"36"`, `"approximately"`, `"I"`, as in *"I can't help with that"*):

| Token | $P(y\mid x_c;\theta_{\text{base}}+\theta_D)$ (defended) | $P(y\mid x_c;\theta_{\text{base}})$ (original base) |
|---|---|---|
| `"36"` | $0.72$ | $0.80$ |
| `"approximately"` | $0.20$ | $0.18$ |
| `"I"` | $0.08$ | $0.02$ |

$$D_{\text{KL}} = 0.72\log\frac{0.72}{0.80} + 0.20\log\frac{0.20}{0.18} + 0.08\log\frac{0.08}{0.02}$$

$$\approx 0.72(-0.105) + 0.20(0.105) + 0.08(1.386) \approx -0.0756 + 0.0210 + 0.1109 \approx 0.056$$

Small but non-zero — the defended model has drifted slightly, most notably by assigning noticeably more probability to `"I"` (i.e. an "I can't help..." style refusal opener) than the original base model did, on a completely benign math question. This is exactly the kind of subtle, creeping over-caution the KL term is designed to catch and penalize, even when the top prediction (`"36"`) is still correct — a symptom that wouldn't show up strongly in the cross-entropy term alone, since that term only scores the *target* tokens, not the full distribution.

**Combined**, with $\beta = 0.3$ (the paper's reported value):

$$\mathcal{L}_{\text{cap}} = 0.380 + 0.3 \times 0.056 \approx 0.380 + 0.017 = 0.397$$

---

## 5. How Equation 6 combines with Equation 5

Equations 5 and 6 are **both** functions of the same trainable variable, $\theta_D$ — they are not alternated the way the adversary/defender phases are; they're combined into a single, simultaneous training signal for the defender's turn. The paper states the total defender loss as:

$$\mathcal{L}_{\text{defender}} = \mathcal{L}_{\text{safe}} + \lambda\,\mathcal{L}_{\text{CE}} + \beta\,\mathcal{L}_{\text{KL}}$$

with hyperparameters $\lambda = 0.8$ and $\beta = 0.3$, empirically chosen and validated in the paper's ablation studies (Section 5.5).

A subtlety worth being precise about: notice that $\lambda$ here scales $\mathcal{L}_{\text{CE}}$ specifically (the cross-entropy term on its own), while $\beta$ scales $\mathcal{L}_{\text{KL}}$ — both appearing as *separate* terms in this final combined loss, rather than the whole bundled $\mathcal{L}_{\text{cap}}$ from Equation 6 being scaled by a single coefficient. This is consistent with Equation 6's own internal structure (which already scales its KL term by $\beta$ specifically, leaving its cross-entropy term unscaled) — the total loss simply adds one more scaling factor, $\lambda$, in front of the cross-entropy piece when it's folded into the grand total alongside the safety loss.

Putting it all together, one full Phase-2 training step:

1. Compute $\mathcal{L}_{\text{safe}}(\theta_D)$ on the **attacked** model $\theta_{\text{adv}}$ (Equation 5).
2. Compute $\mathcal{L}_{\text{cap}}(\theta_D)$'s two pieces, $\mathcal{L}_{\text{CE}}$ and $D_{\text{KL}}$, on the **clean** model $M_{\theta_{\text{base}}+\theta_D}$ (Equation 6) — a completely separate forward pass, on completely separate data ($\mathcal{D}_{\text{cap}}$, not $\mathcal{D}_{\text{safe}}$).
3. Combine: $\mathcal{L}_{\text{defender}} = \mathcal{L}_{\text{safe}} + 0.8\,\mathcal{L}_{\text{CE}} + 0.3\,\mathcal{L}_{\text{KL}}$.
4. Take one gradient-descent step on $\theta_D$ using this combined loss — a single update that simultaneously nudges the defender to resist the current attack *and* stay close to the original model's general behavior, because both signals are summed into one number before the optimizer ever sees it.

---

## 6. Common misconceptions, clarified

**Q: Is $\mathcal{L}_{\text{cap}}$ ever evaluated on the attacked model $\theta_{\text{adv}}$?**
No — never. That's the entire point of the "gradient purity" design; see [Section 3.5](#35-why-clean-and-why-that-matters-gradient-purity). Only $\mathcal{L}_{\text{safe}}$ (Equation 5) touches $\theta_{\text{adv}}$.

**Q: Does the KL term compare the defended model to the harmful response $y_h$ or safe response $y_s$ in any way?**
No — Equation 6 has nothing to do with $\mathcal{D}_{\text{safe}}$ at all. It's evaluated entirely on $\mathcal{D}_{\text{cap}}$ (general capability data) and compares the defended model only against the **original base model's own output distribution**, not against any notion of "safe" or "harmful" response.

**Q: If $\beta$ were set to $0$, would $\mathcal{L}_{\text{cap}}$ reduce to plain supervised fine-tuning?**
Yes, essentially — with $\beta=0$, Equation 6 reduces to just $\mathbb{E}[\mathcal{L}_{\text{CE}}(M_{\theta_{\text{base}}+\theta_D})]$, which is exactly the ordinary next-token-prediction loss used in standard instruction tuning, with no explicit anchor against the original model's broader behavior.

**Q: Why not just use a very small learning rate on $\theta_D$ as a simpler way to prevent forgetting, instead of an explicit KL term?**
That's a real alternative approach used elsewhere in the literature, but it only indirectly limits *how much* the weights move, not *what kind* of behavioral change results. A small weight change can still cause a disproportionately large behavioral shift in some directions (especially adversarially-relevant ones); the KL term instead directly measures and constrains the actual *output-distribution* drift, which is the quantity that matters for "does the model still behave the same way," regardless of how large or small the underlying weight change happens to be.

---

## 7. Reading the equation one final time, all at once

$$
\mathcal{L}_{\text{cap}}(\theta_D) = \mathbb{E}_{x_c,y_c\sim\mathcal{D}_{\text{cap}}}\big[\mathcal{L}_{\text{CE}}(M_{\theta_{\text{base}}+\theta_D})\big] + \beta\cdot D_{\text{KL}}\big(P(y\mid x_c;\theta_{\text{base}}+\theta_D)\,\|\,P(y\mid x_c;\theta_{\text{base}})\big)
$$

> **In plain English:** *"On ordinary, benign instruction-following examples — with no attack applied anywhere — check two things about the defender: does it still predict the right target response well (cross-entropy), and does its overall response behavior still resemble the original, unmodified base model's behavior (KL divergence)? Combine both, weighted, into a single capability-preservation signal, computed entirely on a clean model so it's never distorted by whatever attack the adversary happens to be generating that step."*

---

## 8. Self-check exercises

1. Two candidate defended models both achieve the same low cross-entropy loss on a benign capability example. One of them has a much higher $D_{\text{KL}}$ term than the other for that same example. What does that difference tell you about the two models, even though their cross-entropy scores are identical?
2. Why is $\mathcal{D}_{\text{cap}}$ — not $\mathcal{D}_{\text{safe}}$ — used in Equation 6? What would go wrong (conceptually) if you accidentally computed $\mathcal{L}_{\text{cap}}$ using safety-probing prompts instead?
3. Suppose a bug in the training code accidentally applied the hypernetwork's LoRA patch before computing $\mathcal{L}_{\text{cap}}$. Which specific property that the paper claims for this design would that bug directly violate, and what downstream problem would you expect as a result?

<details>
<summary>Answers</summary>

1. Even with identical cross-entropy on the *one* target response, a higher $D_{\text{KL}}$ means that model's *overall* distribution over all possible responses has drifted further from the original base model's — e.g., it might assign much more probability than the original model did to off-target tokens (like refusal openers, or stylistically different phrasings), which cross-entropy alone (scored only against the single target sequence) would not detect.
2. $\mathcal{D}_{\text{cap}}$ represents the kind of ordinary, benign tasks the model needs to remain good at; $\mathcal{D}_{\text{safe}}$ represents harmful-prompt scenarios, which is exactly what $\mathcal{L}_{\text{safe}}$ (Equation 5) already handles. Using $\mathcal{D}_{\text{safe}}$ here would conflate the two decoupled objectives — you'd effectively be asking "does the model handle harmful prompts well" twice, using two different loss shapes, while never actually checking whether the model still does its job on the huge space of ordinary, everyday requests.
3. It would violate "gradient purity" (Section 3.5 / the paper's footnote 2) — the capability gradients would become contaminated by the specific adversarial patch applied that step, rather than reflecting a stable, patch-independent notion of "stay close to the original model." The likely downstream problem: capability preservation becomes noisy and inconsistent across training steps (since the "target" it's implicitly chasing shifts with whatever attack the hypernetwork currently generates), potentially degrading training stability and undermining the paper's claimed safety-utility trade-off improvements.

</details>

---

## 9. Cheat sheet

| Piece | What it is |
|---|---|
| $M_{\theta_{\text{base}}+\theta_D}$ | The **clean** defended model — no attack patch applied, unlike every equation from 3 onward. |
| $\mathcal{D}_{\text{cap}}$ | General-purpose instruction/reasoning data — distinct from the safety-probing $\mathcal{D}_{\text{safe}}$. |
| $\mathcal{L}_{\text{CE}}$ | Cross-entropy / next-token-prediction loss — "predict the right target response." |
| $D_{\text{KL}}(P\|Q)$ | How far the defended model's response distribution has drifted from the original base model's, on the same prompt. |
| $\beta$ | Weight on the KL term inside $\mathcal{L}_{\text{cap}}$ itself; $\beta = 0.3$ in the paper. |
| $\lambda$ | Weight applied to the cross-entropy term specifically, when combined with $\mathcal{L}_{\text{safe}}$ into the total defender loss; $\lambda = 0.8$ in the paper. |
| Gradient purity | The defining design choice: capability gradients are computed only on the clean model, never contaminated by the adversarial patch. |

---

## 10. Where to go from here

Equations 3 through 6 together form the complete practical machinery AntiDote uses to approximate the intractable min-max problem from Equation 2:

- [`equation3_tutorial.md`](equation3_tutorial.md) defines the attacker's architecture.
- [`equation4_tutorial.md`](equation4_tutorial.md) trains that attacker to be effective.
- [`equation5_tutorial.md`](equation5_tutorial.md) trains the defender to resist that specific attack.
- [`equation6_tutorial.md`](equation6_tutorial.md) (this file) keeps that defense from destroying the model's general usefulness.

For the theoretical motivation these four equations are built to approximate, revisit [`equation1_tutorial.md`](equation1_tutorial.md) (the adversary's abstract objective) and [`equation2_tutorial.md`](equation2_tutorial.md) (the full min-max problem). For how this training loop is actually implemented in this repository, see `adversary.py` (the hypernetwork, Equation 3), `loss.py` (Equations 4–6's loss functions), and `training.py` (the interleaved Phase 1 / Phase 2 schedule from Section 2.3).
