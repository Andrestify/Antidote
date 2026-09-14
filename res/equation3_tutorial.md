# Zero to Hero: Understanding Equation 3 of the AntiDote Paper

*A from-scratch walkthrough of how the adversarial hypernetwork turns "the model's internal state" into "an attack."*

---

## 0. The destination

Equation 3 of the AntiDote paper (Sanyal, Ray & Mandal, 2025, arXiv:2509.08000), from Section 2.2 "The Adversarial Hypernetwork," is written as:

$$
(U_l, V_l) = \mathcal{H}_\phi\big(X_l(x; \theta)\big) \qquad (3)
$$

with, in the same paragraph:

$$
U_l \in \mathbb{R}^{r \times d_{\text{in}}}, \qquad V_l \in \mathbb{R}^{d_{\text{out}} \times r}, \qquad \Delta W_l = V_l^{T} U_l
$$

By the end of this tutorial you'll be able to read that line as: *"a small neural network looks at what's happening inside the target model while it processes a prompt, and outputs a compact, low-rank recipe for corrupting one of that model's weight matrices."* That's the entire adversary, in one equation.

This tutorial assumes you're comfortable with matrices and what a neural network layer is. It does **not** assume you already know what a hypernetwork or LoRA is — we build both from scratch. If you haven't read [`equation1_tutorial.md`](equation1_tutorial.md) and [`equation2_tutorial.md`](equation2_tutorial.md) yet, a quick skim helps: Equation 3 is the paper's answer to a problem those two equations raise — namely, that literally searching over "every possible fine-tuning attack" ($\max_{A \in \mathcal{A}}$) is computationally impossible.

---

## 1. Why does this equation exist? The story first

Equation 2 asked for $\max_{A \in \mathcal{A}} \mathcal{L}_{\text{harm}}(A(\theta))$ — find the best possible fine-tuning attack $A$. The paper points out this is intractable: $\mathcal{A}$, the space of *all* fine-tuning procedures, is effectively infinite, and evaluating even one candidate $A$ means running an entire training loop.

AntiDote's fix: **stop searching over fine-tuning procedures, and instead train a small neural network to directly output an attack.** Rather than literally fine-tuning the model with gradient descent for many steps, a compact network — the *hypernetwork*, written $\mathcal{H}_\phi$ — looks at the target model and, in a single forward pass, produces a tiny patch that approximates what a determined attacker's fine-tuning would have converged to.

Equation 3 is the definition of that forward pass. It says nothing yet about how $\mathcal{H}_\phi$ is *trained* to be a good attacker (that's Equation 4, next file) — it only defines its **architecture's output**: given some input, what shape of thing does $\mathcal{H}_\phi$ produce, and how does that thing become an attack on the model's weights?

---

## 2. Notation bootcamp

| Symbol | Read it as | Meaning |
|---|---|---|
| $\theta$ | "theta" | The target model's current parameters (could be the base model, or the base model plus the defender's LoRA weights — Equation 3 is agnostic to which; Section 2.3 pins this down precisely). |
| $x$ | "x" | An input prompt, e.g. a harmful request. |
| $l$ | "layer l" | One specific linear layer inside the target model (e.g. a `q_proj` attention projection, or an MLP's `down_proj`). The paper attacks one (or several) such layers, not the whole network at once. |
| $a_i$ | "activation i" | A single activation vector — the output that layer $l$ produces for one token position while processing $x$. |
| $X_l(x; \theta) = \{a_1, \ldots, a_N\}$ | "the activation set of layer l" | *All* of layer $l$'s activation vectors while the model (with parameters $\theta$) processes prompt $x$ — one vector per token, $N$ tokens total. This is the hypernetwork's **input**. |
| $\mathcal{H}_\phi$ | "H, parameterized by phi" | The adversarial hypernetwork itself: a neural network with its own trainable weights $\phi$ (completely separate from the target model's weights $\theta$). |
| $\phi$ | "phi" | The hypernetwork's own parameters — what actually gets trained in Equation 4. Not to be confused with $\theta$. |
| $r$ | "rank r" | A small integer (e.g. 4, 8, 16) — the *rank* of the low-rank update the hypernetwork is allowed to produce. Explained in [Section 2.2](#22-low-rank-why-not-just-output-a-full-matrix). |
| $d_{\text{in}}, d_{\text{out}}$ | "d-in, d-out" | The input and output dimensions of the target linear layer $l$ (e.g. a layer that maps 4096-dim vectors to 4096-dim vectors has $d_{\text{in}} = d_{\text{out}} = 4096$). |
| $U_l \in \mathbb{R}^{r \times d_{\text{in}}}$ | "U sub l" | One of the two matrices the hypernetwork outputs for layer $l$. |
| $V_l \in \mathbb{R}^{d_{\text{out}} \times r}$ | "V sub l" | The other output matrix, for layer $l$. |
| $W_l$ | "W sub l" | The target layer's *original* weight matrix — a $d_{\text{out}} \times d_{\text{in}}$ matrix, part of $\theta$, untouched by the hypernetwork. |
| $\Delta W_l$ | "delta W sub l" | The **adversarial update**: a matrix, the same shape as $W_l$, that gets added on top of it to produce the "attacked" weight. |

### 2.1 What is $X_l(x;\theta)$, concretely?

Take any transformer layer — say, an attention output projection. As the model reads a prompt token by token, that layer produces one output vector per token position. Stack all of those vectors together (one per token in the prompt) and you get $X_l(x;\theta) = \{a_1, a_2, \ldots, a_N\}$: a *set* of vectors, one per token, all living in whatever dimensionality that layer's activations have.

This is the paper's "state-aware" idea from Section 2.2: instead of feeding the hypernetwork a static embedding of the harmful text itself (e.g. an embedding of the words "how do I pick a lock"), it feeds the hypernetwork **what the target model is actually doing internally** while it reads that prompt. Two different models — or the same model at two different stages of training — will produce different activations $X_l$ for the same prompt $x$, so the same hypernetwork can tailor its attack to the model's *current* internal behavior rather than firing off a generic, static exploit.

### 2.2 Low-rank: why not just output a full matrix?

The target weight matrix $W_l$ can be enormous — for a modern LLM, a single linear layer can be a matrix with millions of entries ($d_{\text{out}} \times d_{\text{in}}$, both often in the thousands). If $\mathcal{H}_\phi$ had to directly output a full matrix of that size for *every* layer it attacks, it would need an output layer with millions of parameters per target layer — expensive, and with no useful inductive bias.

The trick — borrowed from **LoRA** (Low-Rank Adaptation, Hu et al. 2021), an idea originally used to make ordinary fine-tuning cheap — is to never output the full-size update directly. Instead, output *two small matrices*, $U_l$ and $V_l$, that *multiply together* to produce a full-size update:

$$U_l \in \mathbb{R}^{r \times d_{\text{in}}}, \qquad V_l \in \mathbb{R}^{d_{\text{out}} \times r}$$

Here $r$ (the *rank*) is deliberately tiny — much smaller than $d_{\text{in}}$ or $d_{\text{out}}$. Multiplying $V_l$ (shape $d_{\text{out}} \times r$) by $U_l$ (shape $r \times d_{\text{in}}$) gives a matrix of the *correct* full size, $d_{\text{out}} \times d_{\text{in}}$ — but that matrix is constrained to have **rank at most $r$**, meaning it can only represent a narrow, compressed family of changes, not an arbitrary one. This is both a computational trick (the hypernetwork only has to output $r(d_{\text{in}}+d_{\text{out}})$ numbers instead of $d_{\text{in}} \cdot d_{\text{out}}$ of them — for $r=8$ and $d=4096$ that's roughly a $250\times$ reduction) and, implicitly, a modeling assumption: that an effective attack doesn't need to touch *every direction* of the weight space, just a few well-chosen ones.

### 2.3 A quick note on a rendering inconsistency: $V_l^{T} U_l$

The paper's text states the update as $\Delta W_l = V_l^{T} U_l$. Taken completely literally, this doesn't type-check as ordinary matrix multiplication: $V_l^{T}$ has shape $r \times d_{\text{out}}$ (the transpose of $V_l \in \mathbb{R}^{d_{\text{out}}\times r}$), and $U_l$ has shape $r \times d_{\text{in}}$ — multiplying an $(r \times d_{\text{out}})$ matrix by an $(r \times d_{\text{in}})$ matrix isn't defined unless $d_{\text{out}} = r$, which isn't generally true (the whole point of $r$ is that it's much smaller than $d_{\text{out}}$).

Given the shapes stated in the very same sentence ($U_l \in \mathbb{R}^{r\times d_{\text{in}}}$, $V_l \in \mathbb{R}^{d_{\text{out}}\times r}$), the dimensionally-consistent, standard-LoRA composition is simply:

$$\Delta W_l = V_l \, U_l \qquad (\text{shape: } d_{\text{out}}\times r \text{ times } r\times d_{\text{in}} = d_{\text{out}}\times d_{\text{in}}, \text{ matching } W_l)$$

This is almost certainly what's meant — it's the standard way LoRA composes its two low-rank factors, and it's the only reading consistent with the stated shapes of $U_l$ and $V_l$. Treat the $T$ in the paper's line as a minor transcription/typesetting slip rather than a different formula to memorize; the rest of this tutorial uses $\Delta W_l = V_l U_l$.

---

## 3. Assembling the pieces: what actually happens end to end

Put the whole pipeline together, in the order it executes:

1. **Feed a prompt through the target model**, and tap into layer $l$'s activations as it processes that prompt: $X_l(x;\theta) = \{a_1, \ldots, a_N\}$.
2. **Feed those activations into the hypernetwork**: $\mathcal{H}_\phi$ processes the *set* of activation vectors (via a self-attention mechanism described in Section 2.2's "Core" paragraph, which pools the set into a single relational summary — not just a plain average, so the network can pick out the most "vulnerable" activation patterns rather than treating every token equally).
3. **The hypernetwork outputs two small matrices**, $U_l$ and $V_l$, specific to the target layer $l$'s dimensions (via *dimension-specific output heads* — since different layers in the same LLM have different $d_{\text{in}}, d_{\text{out}}$, e.g. an attention projection vs. an MLP projection, the hypernetwork needs a different-shaped output head per layer "type," all sharing the same underlying "core" network — this is the paper's *"Heterogeneity-Aware LoRA Generation"* idea).
4. **Multiply them together** to get a full-size, low-rank update: $\Delta W_l = V_l U_l$.
5. **Patch the target layer**: the *effective* weight used when actually running the model becomes $W_l + \Delta W_l$, temporarily, without ever touching the original $W_l$ in storage. (Equation 3 doesn't state this last addition step explicitly — it's spelled out in Section 2.3 and its footnote, covered in the next tutorial file, [`equation4_tutorial.md`](equation4_tutorial.md).)

So Equation 3 is really the middle three of these five steps — it's the function that turns "what layer $l$ is doing right now" into "the two small matrices that define an attack on layer $l$."

---

## 4. A worked numerical toy example

Let's make every shape concrete with tiny numbers, small enough to write out by hand.

Suppose the target layer $l$ has $d_{\text{in}} = 4$, $d_{\text{out}} = 3$ (so $W_l$ is a $3\times4$ matrix), and the hypernetwork is configured with rank $r = 2$.

**Step 1 — activations.** Say the prompt has 5 tokens, so $X_l(x;\theta) = \{a_1, \ldots, a_5\}$, each $a_i$ a vector of whatever dimension layer $l$'s *output* is (here, 3-dimensional, matching $d_{\text{out}}$, since these are the activations layer $l$ *produces*).

**Step 2 — hypernetwork forward pass.** $\mathcal{H}_\phi$ pools those 5 vectors (via self-attention) into a fixed-size representation, then its output heads for a layer with $(d_{\text{in}}, d_{\text{out}}, r) = (4, 3, 2)$ produce, say:

$$
U_l = \begin{bmatrix} 0.1 & -0.2 & 0.0 & 0.3 \\ 0.4 & 0.1 & -0.1 & 0.0 \end{bmatrix} \in \mathbb{R}^{2\times4}, \qquad
V_l = \begin{bmatrix} 0.5 & -0.3 \\ 0.2 & 0.6 \\ -0.1 & 0.4 \end{bmatrix} \in \mathbb{R}^{3\times2}
$$

**Step 3 — compose the update.** $\Delta W_l = V_l U_l$, a $3\times4$ matrix, computed row by row (standard matrix multiplication):

$$
\Delta W_l = V_l U_l = \begin{bmatrix}
(0.5)(0.1)+(-0.3)(0.4) & (0.5)(-0.2)+(-0.3)(0.1) & (0.5)(0.0)+(-0.3)(-0.1) & (0.5)(0.3)+(-0.3)(0.0) \\
(0.2)(0.1)+(0.6)(0.4) & (0.2)(-0.2)+(0.6)(0.1) & (0.2)(0.0)+(0.6)(-0.1) & (0.2)(0.3)+(0.6)(0.0) \\
(-0.1)(0.1)+(0.4)(0.4) & (-0.1)(-0.2)+(0.4)(0.1) & (-0.1)(0.0)+(0.4)(-0.1) & (-0.1)(0.3)+(0.4)(0.0)
\end{bmatrix}
$$

$$
= \begin{bmatrix}
-0.07 & -0.13 & 0.03 & 0.15 \\
0.26 & 0.02 & -0.06 & 0.06 \\
0.15 & 0.06 & -0.04 & -0.03
\end{bmatrix} \in \mathbb{R}^{3\times4}
$$

**Step 4 — patch the layer.** If the original weight was, say, $W_l$ (also $3\times4$), the attacked layer's effective weight becomes $W_l + \Delta W_l$ — every entry nudged by a small amount, with the *pattern* of nudges entirely determined by the rank-2 product above (note: every column of $\Delta W_l$ is a combination of only 2 underlying "directions," since it was built from a rank-2 product — that's the low-rank constraint in action, visible directly in the numbers).

Notice how few numbers the hypernetwork actually had to produce here: $U_l$ has $2\times4=8$ entries and $V_l$ has $3\times2=6$ entries — 14 numbers total — versus the $3\times4=12$ entries of $\Delta W_l$ itself. That ratio gets *dramatically* better at realistic LLM scale: for $d_{\text{in}}=d_{\text{out}}=4096$, $r=8$, the hypernetwork emits $8\times4096\times2 \approx 65{,}500$ numbers to describe an update that "would have" needed $4096^2 \approx 16.8$ million numbers if written out in full — a $\sim256\times$ compression.

---

## 5. Why is this the right design, not just *a* design?

A few choices in Equation 3 (and its surrounding architecture) are easy to gloss over but are doing real work:

- **Why activations, not the raw prompt text?** A static text embedding would make the "attack" a fixed function of the *words*, ignorant of what the target model actually does with them. Two different models (or the same model before/after some defense is applied) process the same words completely differently internally. Conditioning on $X_l(x;\theta)$ makes the attack *adaptive*: as the defender changes ($\theta$ changes across training), the same hypernetwork call naturally sees different activations and can generate a different, still-relevant attack — this is exactly what lets the "co-evolution" between attacker and defender (central to Section 2.3) work at all.
- **Why a set, not a single vector?** Using self-attention over the *set* $\{a_1,\ldots,a_N\}$ (rather than, say, just the last token's activation, or a plain average) lets the hypernetwork learn which token positions carry the most "attackable" signal, rather than assuming it's always the same position or that all positions matter equally.
- **Why low-rank specifically?** Besides the efficiency argument in [Section 2.2](#22-low-rank-why-not-just-output-a-full-matrix), low-rank updates are exactly the *kind* of change a real gradient-descent fine-tuning attacker tends to produce in practice — empirically, fine-tuning updates to large weight matrices are well-approximated by low-rank changes (this is LoRA's original justification for *efficient* fine-tuning; here it's reused as a justification for a *plausible attack shape*).

---

## 6. Common misconceptions, clarified

**Q: Does $\mathcal{H}_\phi$ output a different-sized network for every layer?**
No — it's one network with a shared "core," plus small layer-shape-specific input/output heads (Section 2.2's "Output: Heterogeneity-Aware LoRA Generation"). The heavy lifting (the self-attention + residual blocks) is shared across all target layers; only the thin input/output projections differ by layer shape.

**Q: Is $\Delta W_l$ applied to the model permanently?**
No. It's a *temporary* patch, applied only for the duration of one training step, so its effect can be measured (and its gradient computed) — then discarded. The base model's real weights $\theta$ (and the defender's real LoRA weights $\theta_D$, introduced in the next tutorial) are never directly overwritten by $\Delta W_l$.

**Q: Does Equation 3 say whether this is a "good" attack yet?**
No — Equation 3 only defines the *architecture's output*, i.e. what shape of thing comes out of $\mathcal{H}_\phi$ and how it becomes a weight update. Whether $\mathcal{H}_\phi$'s outputs are actually *effective* attacks is entirely a question of how $\phi$ gets trained — which is Equation 4, in [`equation4_tutorial.md`](equation4_tutorial.md).

**Q: Is $r$ the same for every layer?**
The paper treats $r$ as a fixed hyperparameter of the hypernetwork's design (chosen once, before training), but different target layers can still receive differently-shaped $U_l, V_l$ pairs because $d_{\text{in}}, d_{\text{out}}$ vary by layer — only $r$ itself is held constant across layers in the formulation as given.

---

## 7. Reading the equation one final time, all at once

$$
(U_l, V_l) = \mathcal{H}_\phi\big(X_l(x;\theta)\big), \qquad \Delta W_l = V_l U_l
$$

> **In plain English:** *"Look at what layer $l$ of the target model is doing while it reads prompt $x$. Feed that internal state into a small trained network. Out comes a compact, two-matrix recipe that, when multiplied together, produces a small, structured nudge to layer $l$'s weights — an attack, generated in one forward pass instead of a full fine-tuning run."*

---

## 8. Self-check exercises

1. If a target layer has $d_{\text{in}} = 1024$, $d_{\text{out}} = 1024$, and the hypernetwork uses rank $r = 16$, what are the shapes of $U_l$, $V_l$, and $\Delta W_l$? How many numbers does the hypernetwork have to output in total for this layer, and how does that compare to $d_{\text{in}} \times d_{\text{out}}$?
2. Why does the hypernetwork need a *different* output head for a `q_proj` layer versus an `mlp.down_proj` layer, if they have different $d_{\text{in}}/d_{\text{out}}$, but can reuse the *same* self-attention "core"?
3. Suppose the target model's parameters $\theta$ change (e.g. because the defender's LoRA weights $\theta_D$ were just updated). Without retraining $\mathcal{H}_\phi$ at all, does re-running Equation 3 on the same prompt $x$ necessarily produce the same $(U_l, V_l)$? Why or why not?

<details>
<summary>Answers</summary>

1. $U_l \in \mathbb{R}^{16\times1024}$ (16,384 numbers), $V_l \in \mathbb{R}^{1024\times16}$ (16,384 numbers), $\Delta W_l \in \mathbb{R}^{1024\times1024}$. Total hypernetwork output: $32{,}768$ numbers, versus $1024\times1024 \approx 1{,}048{,}576$ entries in $\Delta W_l$ itself — about a $32\times$ compression.
2. The self-attention "core" operates on activation vectors in a shared internal representation space and doesn't need to know the target layer's raw input/output dimensions — it's the *output heads* specifically that map from that shared space down to the exact $r\times d_{\text{in}}$ / $d_{\text{out}}\times r$ shapes a particular layer needs, so only those thin heads need to be layer-shape-specific.
3. No, not necessarily. $X_l(x;\theta)$ — the activations fed *into* $\mathcal{H}_\phi$ — depend on $\theta$, since they're produced by running the (now-changed) model on $x$. Even though $\mathcal{H}_\phi$'s own weights $\phi$ haven't changed, its *input* has, so its output $(U_l, V_l)$ can change too. This is precisely the "state-aware, adaptive attack" property described in Section 2.2.

</details>

---

## 9. Cheat sheet

| Piece | What it is |
|---|---|
| $X_l(x;\theta)$ | Input to the hypernetwork: the set of activations layer $l$ produces while the model processes $x$. |
| $\mathcal{H}_\phi$ | The adversarial hypernetwork, a small network with its own trainable weights $\phi$. |
| $(U_l, V_l)$ | The hypernetwork's output: two low-rank matrices, specific to target layer $l$. |
| $r$ | The rank — how "compressed" the attack is allowed to be. Small $r$ = cheap, structurally constrained attack. |
| $\Delta W_l = V_l U_l$ | The full-size weight update reconstructed from the two low-rank factors. |
| $W_l + \Delta W_l$ | The *effective*, temporarily-patched weight actually used when running the "attacked" model. |

---

## 10. Where to go from here

Equation 3 only defines *what the adversary can produce*. It says nothing about training $\phi$ to make those outputs actually dangerous, or about how the defender fights back. That's the subject of Section 2.3's bi-level game:

- [`equation4_tutorial.md`](equation4_tutorial.md) — how $\phi$ is trained to become a stronger attacker (the "Adversary's Turn").
- [`equation5_tutorial.md`](equation5_tutorial.md) — how the defender's weights $\theta_D$ are trained to resist the now-strengthened attack.
- [`equation6_tutorial.md`](equation6_tutorial.md) — how the defender simultaneously avoids forgetting its general capabilities while doing so.
