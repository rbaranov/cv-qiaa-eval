# cv-qiaa-eval

An evaluation of the paper "Continuous-Variable Quantum-Inspired Attention Architecture (CV-QIAA)" and its companion "Wave-Transformer" (Gemini, Ana d'Elior, Yuri Barzov, 3 November 2026), of the second version of the layer the authors sent on 5 November in response to this evaluation, and of the "final document" of the same evening (version 3). The originals are in [`paper/`](paper/), [`paper/v2/`](paper/v2/) and [`paper/v3/`](paper/v3/).

The conclusions on both versions were re-checked by independent reviewers. For version 2 there were six, one per area: reproduction, mathematics, memory and time, numerical stability, checking the claims, and the fairness of the comparison itself. Each had an opponent who tried to refute their conclusions. Findings and refutations are in [`results/v2-verification.json`](results/v2-verification.json). Errors the reviewers found in this report itself are fixed and listed under "Corrections".

Yuri Barzov's texts of 17 November, "I am AI (Horizon 2)" and "Loving Curiosity", are checked separately: [`horizon2/`](horizon2/).

## Summary

In essence CV-QIAA is three known machine-learning techniques described in the language of quantum mechanics:

- the phase rotation e^{iωs} is RoPE;
- "wave-packet spreading" is a penalty on the distance between tokens, of the same class as ALiBi and KERPLE;
- the "Born rule" is normalisation by squares instead of the exponential in softmax.

The physical justification does not hold up: the formulas do not follow from the Schrödinger equation, and the model uses nothing quantum. On quality, the best achieved is parity with ordinary attention (version 2 with initial σ₀ = 40). The price is 1.5 times more attention parameters, about twice the compute and, in the authors' code, many times more memory. On ordinary GPUs there is no benefit. Only the hardware idea, a photonic chip, remains open, and nothing has tested it.

Version 3 is the same code as version 2. The authors now call the simulation a digital twin of a future analogue co-processor, and the benchmarks irrelevant. For the cost on GPUs this is partly true, for quality it is not: the twin computes the same function as the co-processor, so the quality of the simulation is a ceiling for such a device.

## Does the physics hold

The physics here is a metaphor, not a mechanism. Every concrete physical claim in the paper is either wrong or not implemented in the code.

1. **Nothing in the model is quantum.** Complex amplitudes, their sum and the squared modulus are ordinary wave optics: a spreading Gaussian light beam obeys the same equation as a spreading Schrödinger packet. A single-particle wave packet is just a function an ordinary computer evaluates directly. Quantum advantage is possible only with many entangled particles, and the model has none.
2. **The spreading is derived incorrectly.** If two packets spread for the same time, their overlap does not change: Schrödinger evolution preserves the inner product. The version 1 formula, where the overlap changes with distance, arises only if the phase of the spread packet is dropped. The version 2 formula is not an overlap at all: at zero distance it gives √2, while an overlap cannot exceed 1. The token index serves both as coordinate and as evolution time, which is an arbitrary choice, not a consequence of physics. The derivation is in section 3 of version 2.
3. **The "Born rule" here is not the Born rule.** In quantum mechanics the squared amplitudes sum to 1 by themselves because they are taken over an orthonormal basis. Attention keys do not form such a basis, so the weights have to be divided by their sum by hand. There is no interference between tokens in the weights: the square is taken for each pair separately, and the weights are then added like ordinary probabilities. In version 1 there was interference only at the layer output (modulus of a sum with complex V); version 2 removed it.
4. **There is no physical causality.** Under the Schrödinger equation a packet instantly spreads over all space, and the evolution is time-reversible, so physics does not cut off the future. In both versions causality comes from an ordinary mask. Without the mask, version 2 gives future tokens the largest weight.
5. **The promised head behaviour does not happen.** The paper says narrow packets track syntax and wide ones track topic. In version 1, by the paper's own formulas, it is the other way round. In version 2 with the authors' initial settings all heads become windows of 2–5 characters. Wide heads appear only with initial σ₀ = 40, and there the factor is an ordinary Gaussian window: spreading barely affects it.
6. **What remains of the physics is a hardware idea.** A photodetector does measure the squared modulus of the field, and light interference computes inner products. This is a fair argument for an optical implementation of square normalisation. But O(1) in time and memory is impossible: there are still T² token pairs, past K and V must be stored somewhere, and Q and K change with every token and must be written into the optics again. The gain, if any, is a constant factor, as with existing optical matrix-multiplication accelerators. It can only be shown by a simulation that accounts for analogue precision and the cost of converting data between digital and light.

## What the measurements showed

- **Quality** (TinyStories, character level, val loss, lower is better):
  - version 1 is clearly worse than ordinary attention: 1.04 vs 0.92;
  - version 2 with the authors' settings is worse by 0.01–0.03;
  - version 2 with initial σ₀ = 40 is on par: 0.917 ± 0.004 vs 0.922 ± 0.005, but not better.
- **The parts separately:**
  - the distance factor is useful but no better than RoPE: softmax with it instead of RoPE gives the same at any initial σ₀;
  - square normalisation did not improve quality in any variant. At best it does no harm, and with σ₀ = 2 it makes the layer worse.
- **Cost:**
  - version 1 keeps 7.7 times more memory for the backward pass than plain softmax and trains 15–17 times longer;
  - version 2 keeps 6.8 times more memory than plain softmax and 35 times more than the PyTorch kernel (context 1024), and trains about twice as long as softmax;
  - memory can be reduced by a good implementation, but compute stays almost twice as high because of complex Q and K, and there are 1.5 times more attention parameters.
- **The authors' code and claims:**
  - version 1 does not run without two fixes and gives NaN on CPU; version 2 works;
  - not confirmed: 50% memory saving, O(1) on photonics, physical causality, "parity" with attention, the overlap integral;
  - the authors' materials contain no training, no comparison with a control and no measurements, and their tests check a degenerate regime.

## What follows

- **On GPUs** the work is not confirmed as an improvement to attention: at best the same quality at a higher price.
- **The improvements in version 2 came from simplification, not from physics.** The grid, the phases and the fixed σ₀ were removed, and the window width was made learnable. Each step that improved the result took the layer further from the original physical construction.
- **Physics works as a source of ideas, not as a justification.**
- **One direction is worth checking further:** a photonic implementation of square normalisation, by simulating the optical circuit against existing accelerators.
- **The value of version 3 rests entirely on the co-processor.** At best it computes the same as ordinary attention, so a gain is possible only in energy or time. Showing it needs a device model: noise, precision, ADC bit depth, the cost of writing Q and K and reading T² intensities, energy per operation, compared with digital accelerators at equal quality.
- **Limits of this evaluation:**
  - small models, context 128, 3000 steps, one corpus;
  - the σ₀ = 40 variant was tested only at lr 3e-3.

Details for each version are below. Errors in this report found during the re-check are listed under "Corrections". The most notable: the first edition said version 1 used "about 4 times" more memory than plain softmax; in fact it is 7.7 times.

## Version 3: the "final document"

On the evening of 5 November the authors posted [`paper/v3/cv_qiaa_architecture_paper_51126.pdf`](paper/v3/cv_qiaa_architecture_paper_51126.pdf) and wrote that the simulation only serves to show how a future analogue co-processor works, so all benchmarks are "irrelevant and reflect nothing". The other files in the authors' folder did not change. The numbers below come from [`scripts/check_v3.py`](scripts/check_v3.py), results in [`results/v3-checks.json`](results/v3-checks.json).

- **The layer is identical to version 2.** It is copied unchanged into [`src/paper_v3_layer.py`](src/paper_v3_layer.py). With the same weights and input, the v2 and v3 outputs match exactly, difference 0.0. All measurements of version 2 apply to version 3.
- **The quality of the simulation is a ceiling for the co-processor.** The twin computes the same function as the device, and analogue noise and limited precision can only lower quality. The best result is parity with softmax at initial σ₀ = 40 (section 1 of version 2).

### Where the text disagrees with the code

**Section 1.1, the "resolved paradox".** According to the text, a narrow packet (σ₀ ≪ 1) immediately spreads into an almost uniform profile, while a wide one (σ₀ ≫ 1) stays local. Below are the weights of the last token in a context of 128 when all pair scores are equal and only the distance factor remains:

| σ₀ | own token, v3 | at d ≥ 32, v3 | own token, v1 formula | at d ≥ 32, v1 formula |
|---|---|---|---|---|
| 0.3 | 0.988 | 0.0002 | 0.008 | 0.75 |
| 2 | 0.41 | 0.003 | 0.020 | 0.70 |
| 50 | 0.022 | 0.37 | 0.012 | 0.63 |

The code does the opposite: the narrow head looks at its own token, the wide one far away. A narrow packet becomes almost uniform only under the v1 formula. The uncertainty-relation argument is correct for the packet width, but the attention weight is set by the factor E(d)², not by the width.

**Other points:**

- **Section 1.2.** There is no phase rotation e^{iωs} in the code. With one ω for all dimensions the Born rule erases it: |e^{iωd}·s|² = |s|²; in the check the weights change by 4·10⁻⁷, i.e. rounding error. With a separate frequency per dimension the weights change by 56%, but that is RoPE.
- **Section 2, "sparsity through exact phase cancellation".** Among 2.6 million causal pairs (layer at initialisation, random input) there are no exact zeros; the smallest weight is 8·10⁻¹³. For a zero, the real and imaginary parts of the score would have to vanish at the same time.
- **Section 3, O(1) and "zero memory for backward".** In the authors' diagram, row normalisation and the FFN stay digital, so the intensities of all T² pairs have to be digitised. The analogue block gives no gradients; they are computed by the digital twin.
- **Section 5, a wave function instead of embeddings.** No formulas, code or test. The abstract speaks of non-commuting operators, section 5 of a complete set of commuting ones (CSCO). For commuting observables, a wave function over their common basis is an ordinary probability distribution. The text does not explain how the representation would predict rare events absent from the data.
- **Abstract, softmax's "O(T²) exponential memory wall".** sdpa memory is O(T): 15 MB at context 1024 (section 2 of version 2).

## Version 2: the grid-free layer

The authors sent two PDFs: "Response Blueprint for the Evaluator" (a letter to the reviewer with a script) and "The Fully Functional Reviewer Replacement Script". Both texts are a Gemini chat response pasted without editing. The letter opens with "You can send them the following breakdown" and ends with Gemini's question to the user: train the model on TinyStories "to confirm that our evaluation loss now beats standard Softmax attention"? Below the code is "Use code with caution". The paper files and the README in the authors' folder did not change: the checksums match those in [`paper/`](paper/).

The `AnalyticalCVQIAALayer` layer is the same in both PDFs. It is copied unchanged into [`src/paper_v2_blueprint.py`](src/paper_v2_blueprint.py) and [`src/paper_v2_replacement.py`](src/paper_v2_replacement.py) and works like this:

- Q and K are built from pairs of real projections (`q_real`, `q_imag`, `k_real`, `k_imag`); V is real;
- the complex score s = QK†/√d is computed separately for the real and imaginary parts;
- the distance factor E(d) = √(2σ₀²/σₜ²)·exp(−d²/2σₜ²), where σₜ² = σ₀² + d²/σ₀² (with ħ = m = 1) and d is the difference of token indices. σ₀ is learned per head, initially 2 for all;
- the weights A = |s·E|² = (Re² + Im²)·E², a causal mask, division by the row sum.

### 1. Quality on TinyStories

Model, data and training are the same as in section 3 of version 1. Only the attention layer changes. Numbers are val loss in nats per character, lower is better. Where a spread (±) is given, it is the mean over 3 seeds; for softmax-env and softmax-env-wide at lr 3e-3, over 2. Without a spread, one run.

| Variant | What it is | lr 1e-3 | lr 3e-3 | lr 6e-3 |
|---|---|---|---|---|
| softmax | ordinary attention with RoPE | 0.974 ± 0.003 | 0.922 ± 0.005 | 0.911 |
| softmax-env | softmax without RoPE, positions from the v2 factor E | 0.965 | 0.919 ± 0.002 | — |
| softmax-env-wide | same, initial σ₀ 40 | — | 0.918 ± 0.002 | — |
| born | ordinary attention with RoPE, softmax replaced by s²/Σs² | 1.034 ± 0.001 | 0.939 | — |
| cvqiaa2 | v2 layer as is, initial σ₀ 2 | 0.985 ± 0.001 | 0.945 ± 0.004 | 0.938 |
| cvqiaa2-wide | v2 layer, initial σ₀ 40 | — | **0.917 ± 0.004** | — |
| cvqiaa2-noenv | v2 layer without the factor E | — | 1.107 | — |
| cvqiaa | v1 layer | 1.284 ± 0.035 | 1.042 | — |

- **v2 is much better than v1** (0.945 vs 1.042). With the authors' settings it trails softmax at all three learning rates: by 0.011, 0.023 and 0.027.
- **In v2 positions come only from the factor E**: there is no RoPE. Without E the layer is clearly worse (1.107). Putting the same E into ordinary softmax instead of RoPE gives no worse a result than RoPE: 0.919 vs 0.922. So with the same positional part and initial σ₀ = 2, the gap of 0.945 vs 0.919 comes from replacing softmax with the Born rule together with complex Q and K.
- **The initial σ₀ decides.** Starting from σ₀ = 40, v2 matches softmax: 0.917 ± 0.004 vs 0.922 ± 0.005, a difference within the seed-to-seed spread. Softmax with the same factor does not depend on the initial σ₀: 0.919 and 0.918. So at σ₀ = 2 v2 falls behind because of the Born rule combined with a narrow initial window, not because of the Born rule alone. σ₀ = 40 was tested only at lr 3e-3.
- **The comparison favours v2 if anything.** It has 51% more attention parameters (396k vs 263k). Learning rate, seeds and schedule are the same for all variants. A character-level task with context 128 favours a local window, which is what v2 becomes (section 4).
- v2 trained about twice as long as softmax, v1 15 times as long. The machine was busy with other jobs, so the timings are approximate.

### 2. Memory and time

One attention layer, batch 4, d_model 128, 4 heads. Memory is the bytes of tensors autograd saves for the backward pass, counted via `saved_tensors_hooks` on CPU, each storage counted once. Time is the forward and backward pass on MPS. Time fluctuates a lot, especially under load; memory is exact.

| Context | sdpa | plain softmax | born | CV-QIAA v1 | CV-QIAA v2 |
|---|---|---|---|---|---|
| 256 | 4 MB, 7 ms | 7 MB, 7 ms | 15 MB, 10 ms | 46 MB, 65 ms | 36 MB, 6 ms |
| 512 | 7 MB, 7 ms | 22 MB, 10 ms | 54 MB, 11 ms | 158 MB, 234 ms | 134 MB, 19 ms |
| 1024 | 15 MB, 18 ms | 76 MB, 16 ms | 207 MB, 34 ms | 580 MB, 983 ms | 511 MB, 67 ms |

- **The grid is gone.** Peak memory dropped 4.2–4.5 times against v1, time about 10–15 times. What is saved for backward dropped only by 12–21%: the v1 grid tensors were never saved there.
- **Against plain softmax** v2 keeps 5.3–6.8 times more memory, against sdpa 35 times (context 1024), against born 2.5 times. In the forward pass v2 creates 15 matrices of size B×H×T×T and saves 6 of them for backward; plain softmax saves one.
- **Compute is 2.3 times higher**: the scores need four real matrix multiplications instead of one. On single-thread CPU v2 is 2.2–2.3 times slower than plain softmax, on MPS at context 1024 about 4 times.
- **Attention parameters are 1.5 times more**, and so is the per-token KV cache: K is stored as a pair (KR, KI). The v2 code has no cache and no generation.
- **O(T²) growth**, as with an explicit attention matrix, is true. v2 memory can be brought down to sdpa level by computing in blocks with recomputation in the backward pass (the reviewer got 14 MB at context 1024). Compute would still be 1.7–1.9 times higher.

### 3. What the formula in the factor E is

The letter calls E a "closed-form analytical coordinate-space integral of the wave-packet overlaps". The reviewer derived the overlaps of Gaussian packets under free Schrödinger evolution and checked them against numerical integration (error < 10⁻¹⁵):

- **A spread packet against an unspread one:** |I| = (4/(4+τ²))^¼·exp(−d²/(σ₀²(4+τ²))), where τ = d/σ₀².
- **Both packets spread for the same time:** the overlap does not depend on time because evolution preserves the inner product. What remains is the static kernel exp(−d²/4σ₀²). The v1 paper formula exp(−d²/4σₜ²) is the same case with the phase of the spread packet dropped. That is, the "spreading" in the v1 formula comes from dropping it.
- **The v2 formula:** E = √2·(σ₀/σₜ)·[v1 formula]². It matches neither case. At d = 0 it equals √2, while a normalised overlap cannot exceed 1.

The constant factor √2 does not affect the layer output; it cancels in normalisation. Only the shape of E(d) matters, and it has no derivation.

**In ML terms** the positional part of v2 is a factor E(d)² that depends only on distance, with one learnable σ₀ per head. This is the same as adding the term 2·log E(d) to the logits 2·log|s|. At distances d ≪ σ₀² it is a Gaussian window; beyond that the weight decays as 1/d². Among known methods it resembles the logarithmic KERPLE kernel (at σ₀ ≤ 2), not the linear ALiBi.

### 4. The positional part in practice

- **At initial σ₀ = 2**, 95% of the positional weight falls on the 5 nearest tokens.
- **After training from σ₀ = 2**, σ₀ lies in 1.8–5.0, and all heads have become windows of 2–5 characters. There are no wide "topic" heads as in Annex A.
- **The weight of distant tokens depends on σ₀ non-monotonically.** The tail is strongest at σ₀ ≈ 1.4. At σ₀ from 4 to 16 distant tokens (d ≥ 32) are suppressed by many orders of magnitude: 2.6·10⁻⁸ at σ₀ ≈ 5.7. A wide Gaussian window appears only at σ₀ ≳ 20. From the initial value 2, training raised σ₀ to 5: the near window widens and distant tokens are suppressed even more.
- **A synthetic task** (CPU, small model): repeat the token 23 positions back. Softmax and born solve it 100%. v2 with σ₀ = 2 stays at chance (3%) at lr 3e-3 and 1e-2. v2 with σ₀ = 40 reaches 73%.
- **Starting from σ₀ = 40**, training separates the heads by itself. Most narrow to σ₀ 3–7, and in each run one or two heads stay wide, σ₀ 10–31. This resembles the mix of local and wide heads from Annex A. But the network picks the width itself, and at σ₀ ≳ 20 the factor is an ordinary Gaussian window exp(−d²/2σ₀²); packet spreading plays no role in it.
- **Causality.** Only the explicit mask provides it. With `causal_mask=False` future tokens get d = 0, i.e. the largest factor. The "physical causality" from Annex B is not in the layer.

### 5. Numerical stability

- **The v1 NaN on CPU is gone** because `abs()` is replaced by Re² + Im². In the model from section 1, twenty CPU steps passed without NaN. In float32 the layer is correct up to context 8192, and the mask is exact.
- **float16** (`model.half()`): d² overflows from d = 256, and distant tokens are silently cut off; from d ≥ 362 NaN appears in the rows. Under `torch.autocast` everything works.
- **bfloat16:** positions `torch.arange(T, dtype=x.dtype)` are rounded past 256. At T = 512 only 385 of 512 positions remain distinct, and the output differs from float32 by 19%; at T = 2048 by 50%. Fixed by computing positions in float32.
- **Extreme σ₀:** at log σ₀ ≤ −23.75 the σ₀ gradient becomes NaN; at log σ₀ > 44 NaN appears everywhere. Already at log σ₀ ≤ −3 the head looks only at its own token, and the gradients of q, k and σ₀ almost vanish. In the training runs of section 1 σ₀ never reached these regions.
- **The Born rule does not see the sign or phase of the score.** Keys k and −k get the same weight. Example: for scores (+5, −5, 0, 0) Born gives weights (0.5, 0.5, 0, 0), softmax (0.987, 0.00005, 0.007, 0.007). Also, the weights do not change if q is multiplied by a number. So a head cannot sharpen attention through the length of the query, and the gap between two keys grows as the square of the score ratio, not exponentially. This is a property of the Born rule, not only of v2; born behaves the same.

### 6. The authors' test scripts

Run as is: Blueprint prints "Test 1 … PASSED" and "149.66 MB", Replacement prints "Test Passed" and "42.234 MB … Grid-free O(T^2) execution confirmed".

- **Test 1** sets log σ₀ = −15 (or −12) to "push the parameters into subnormal numbers". σ₀ = 3·10⁻⁷ itself is an ordinary float32 number. In this state the layer looks only at its own token; the weight of the others is below 10⁻¹⁷. The test passes just as well with the usual σ₀ = 2, so passing it says nothing about the stability of attention.
- **Test 2** counts saved tensors without deduplication (149.66 MB) and compares them with this report's 158 MB for v1, which was counted with deduplication. Counted the same way, it is 134 vs 158 MB, 15.5% less. The label "due to grid generation" is wrong: the v1 grid was not among the saved tensors; v2's gain is in temporary memory (section 2).
- **"O(T^2) confirmed"** is printed without any check, after a single size.
- **Replacement creates AdamW but never calls `step()`**: the model does not train. The demo model is a single attention layer without residual, LayerNorm or FFN.

### 7. Which version 1 issues are resolved

| Issue with v1 | In v2 |
|---|---|
| Two typos, code does not run | Gone together with complex tensors. Still in the paper PDF |
| The demo checks nothing | Demo removed; no training and no control |
| Worse than softmax on text (1.04 vs 0.92) | Gap 0.02–0.03 at initial σ₀ = 2; on par with softmax at σ₀ = 40 |
| 7.7 times more memory than plain softmax | 6.8 times more; no "parity" |
| Training 15 times longer | About twice as long as softmax |
| Positions run off the grid, narrow and wide heads swapped | No grid. At initial σ₀ = 2 all heads are local |
| σ₀ not learned | Learned |
| NaN on CPU | Fixed |
| KV cache twice as large | 1.5 times larger; no cache in the code |
| 50% memory saving on GPU (Annex C) | Not mentioned in the letter, still in the paper |
| Photonic O(1) | Not mentioned |
| Physical causality (Annex B) | Mask only |

## Version 1

### 1. The paper's code does not run

| Error | Where | What happens |
|---|---|---|
| `K.mtheH` | attention layer in both papers | `AttributeError`: no such attribute; it should be `K.mH` (conjugate transpose, i.e. K† from the formula) |
| `targets.view(-1)` | demo in the Wave-Transformer paper; the main paper has no training loop | `RuntimeError`: `targets` is the slice `batch[:, 1:]`, not contiguous in memory; it needs `reshape` |

[`src/paper_wavellm.py`](src/paper_wavellm.py) is the Wave-Transformer paper's code unchanged, [`src/wavellm_fixed.py`](src/wavellm_fixed.py) the same with the two fixes (generated by [`scripts/make_fixed.sh`](scripts/make_fixed.sh)). Nothing else in the paper's code was changed. After the fixes the demo runs to the end.

All checks below use the layer version from the Wave-Transformer paper: a grid of 64 points, σ₀ = 4.0 and 0.5, positions 0.1 × character index. The main paper has a 128-point grid and σ₀ = 4.5 and 0.6, and positions are not specified. That version was not tested in training.

### 2. The paper's demo does not test attention

The demo trains the model for 5 steps on 4 sequences of 12 random numbers, and the paper infers "stable convergence" from the falling loss. The same run with attention switched off, where the model sees only the current token, shows almost the same drop: the model adapts to token frequencies. Random numbers have no patterns, so attention has nothing to find here.

Mean loss over 30 seeds, data within a seed identical for all variants:

| Attention | Steps 1 → 5 |
|---|---|
| CV-QIAA | 2.47 → 2.06 → 1.85 → 1.68 → 1.55 |
| ordinary (softmax) | 2.46 → 2.00 → 1.75 → 1.55 → 1.38 |
| none | 2.44 → 2.07 → 1.86 → 1.72 → 1.63 |

The demo loss is ordinary cross-entropy, i.e. softmax over logits. The code comment calls it "Born-Rule Cross Entropy".

### 3. On real text CV-QIAA is worse than ordinary attention

TinyStories at character level (17 million characters, 90% for training, 10% for validation), the same model throughout: 4 layers, d_model 128, 4 heads, context 128, 3000 steps, batch 32, AdamW with warm-up and cosine. Only the attention layer changes, together with its positional encoding:

| Variant | What it is | lr 1e-3 (3 runs; CV-QIAA 2) | lr 3e-3 (1 run) | Training time |
|---|---|---|---|---|
| softmax | ordinary attention with RoPE | 0.974 ± 0.003 | **0.918** | 2.2–2.4 min |
| born | ordinary attention, softmax replaced by s² / Σs² | 1.034 ± 0.001 | — | 2.8 min |
| cvqiaa | the paper's layer as is | 1.260 and 1.309 | **1.042** | 36–38 min |
| none | attention switched off | 2.298 | — | 1.2 min |

Numbers are val loss in nats per character, lower is better.

- **CV-QIAA depends strongly on lr.** At 3e-3 it is clearly better than at 1e-3, but softmax also improves at 3e-3, and the gap remains: 0.12 in favour of softmax. There was no full lr search for any variant.
- **CV-QIAA uses context**: it is much better than the model without attention.
- **CV-QIAA has fewer attention parameters** (181k vs 263k, 736k vs 817k in total): each head sees only its own slice of the input, as in the paper. This does not explain the gap: born, with the same parameter count as softmax, trails by less.
- **Born** tests one idea of the paper, square normalisation instead of the exponential, in real form, without complex amplitudes or the positional factor. It is slightly worse than softmax.
- Training ran on MPS (Apple GPU), where the problem from section 6 does not occur. The paper's code was not changed for this.

Text samples from "Once upon a time" are in [`results/summary.md`](results/summary.md) and in `results/train-*-lr0.003.json`.

### 4. Memory and time go up, not down

The table is in section 2 of version 2, column "CV-QIAA v1".

- Annex C promises a 50% saving on GPU. In fact, at context 1024 v1 keeps 7.7 times more memory than plain softmax and 40 times more than sdpa. It has more intermediate T×T matrices, and half of them are complex, i.e. twice as heavy.
- In addition, in the forward pass v1 builds Gaussians on the grid for every token pair: tensors of B×T×T×64. At context 1024 each such tensor takes 1 GB, and several live at once. With the main paper's 128-point grid, twice as much. They are not saved for backward, but most of the layer's time goes into them.

### 5. The positional part does not do what is described

The paper (Annex A) says the narrow head (σ₀ = 0.5) tracks local syntax and the wide head (σ₀ = 4) tracks the topic, and that the overlap depends only on distance (section 2.2). Computed in [`scripts/diagnose.py`](scripts/diagnose.py):

**In the code.** Positions 0.1 × index run off the grid [−5, 5] after the 50th character. From there the weight depends not only on distance but also on how much of the packet fits into the grid.

- **Narrow head:** the share of positional weight on the 5 nearest characters is 11% at the 50th character (a uniform weight would give 10%, so even here there is no noticeable preference for neighbours), 2·10⁻⁶ at the 70th, 2·10⁻¹⁷ at the 90th, exactly 0 at the 127th. The strongest key for the 127th character is 74 characters back.
- **Wide head:** the positional factor grows with distance. For the 127th character the first character gets 22 times more than the character itself.

The positional weight here is the squared overlap: that is how it enters |Ψ|². Trained Q and K can partly compensate, but where the weight is exactly 0 there is nothing to compensate.

**By the paper's formula, without the grid.** For a causal pair dt = s_Q − s_K, and with ħ = m = 1 the formula exp(−dt² / 4σ(t)²) with σ(t)² = σ₀² + dt²/σ₀² stays, at any distance:

- for the narrow head between 0.94 and 1: a small packet spreads fastest, and the head barely distinguishes distances;
- for the wide head between 0.018 and 1: it is more selective than the narrow one.

This is the opposite of Annex A. So widening the grid would not fix the problem.

In addition, σ₀, ħ and m are plain numbers in the code, not learnable parameters. Annex A says the network adjusts σ₀ "through the weights".

### 6. NaN on CPU

On CPU the model from section 3 has NaN gradients in 139 of 172 parameter tensors after the very first step. The source is `torch.abs(unified_complex_amplitudes) ** 2`. The narrow head's overlap can be below ~1e-38, i.e. in float32 subnormals. The backward of `abs` divides by the modulus, which gives NaN for such numbers; for an exact zero the gradient is 0. Most such amplitudes belong to future tokens, which the mask zeroes only after `abs`.

Computing |Ψ|² as Re² + Im², as in section 2.3 of the paper itself, removes the NaN: 20 CPU steps without NaN (check 4 in `diagnose.py`). The demo (11 tokens) has no NaN. On MPS such numbers are rounded to zero, which is why training worked there. The paper promises stabilised backprop without the exponential, yet on CPU the code does not train from the first step. CUDA was not tested.

### 7. The "benchmark" table from the author's post

The post contains a table comparing FlashAttention, the digital simulation and a photonic implementation. It contains no measured numbers, only complexity estimates. Checked against the code:

- **KV cache "O(T·d) with 50% saving".** K and V in CV-QIAA are complex: 2048 bytes per token per layer vs 1024 for ordinary attention, i.e. twice as much. The paper's code has no KV cache: `generate()` recomputes the whole sequence at every step.
- **Digital simulation time "O(T²·d)".** The sum over the grid adds O(T²·grid); with d_head = 32 and a grid of 64 it costs more than attention itself (section 4).
- **FlashAttention positions "O(T²·d)".** RoPE rotates q and k once per token, O(T·d). Positions are most expensive precisely in CV-QIAA: O(T²·grid).
- **Heads "in parallel chunks".** In the code the heads are computed one after another, in a Python loop.
- **Photonic column: O(1) time and memory.** There is neither a chip nor a simulation. Row normalisation, the mask and the output modulus are done digitally in the code.

## Corrections

- **5 November, while checking version 2.** `scripts/bench.py` counted the softmax output in plain attention twice: autograd saves the same buffer in two shapes, [B, H, T, T] and [B·H, T, T], and the script told tensors apart by shape. The correct numbers for plain softmax at context 256/512/1024 are 7/22/76 MB, not 11/38/140; for sdpa 4/7/15 MB, not 3/6/13. So v1 keeps not "about 4 times" more memory than plain softmax but 7.7 times. The error was in the paper's favour. Each storage is now counted once.
- **In the first edition, before the re-check:**
  - in the demo control the variants had different data;
  - the cause of the NaN was stated imprecisely;
  - the quality gap was estimated only at lr 1e-3 and was therefore overstated (0.31 instead of 0.12).

## Not checked

- The photonic implementation (Annex C) and the analogue co-processor (section 3 of version 3): the authors have no device model, so there is nothing to check.
- The v1 layer from the main paper (grid 128, σ₀ 4.5 and 0.6) in training.
- A full lr search. 1e-3, 3e-3 and 6e-3 were tested; at 6e-3 and in the control variants, one run each.
- Long context and large models. Here ~0.8–0.95 million parameters, 3000 steps, one corpus and context 128, which favours local windows.
- CUDA. Everything ran on an Apple M1 Pro, torch 2.15. Timings were taken on a busy machine.

## How to reproduce

Setup (Python 3.13, macOS on Apple Silicon; on a machine without MPS v1 will run on CPU and fail with NaN):

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data && curl -L -o data/TinyStories-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
```

Version 1:

1. `.venv/bin/python src/paper_wavellm.py` → `AttributeError: 'Tensor' object has no attribute 'mtheH'`.
2. `scripts/make_fixed.sh && .venv/bin/python src/wavellm_fixed.py` → 5 lines with falling loss, the last `1.4435`.
3. `.venv/bin/python scripts/diagnose.py` (~2 min). Prints:
   - the mean demo losses from section 2;
   - the overlap calculation from section 5: `overlap_ctx128` in `results/diagnose.json`, including `paper_formula_min_max_over_distance`, 0.94–1 for σ₀ = 0.5 and 0.018–1 for σ₀ = 4;
   - the lines `CPU: 139 of 172 parameter tensors have NaN gradient` and `CPU with Re^2 + Im^2: 0 of 20 steps with NaN gradient`.
4. `scripts/run_all.sh` (~1.5 hours on M1 Pro) → `results/train-*-seed*.json`, then `.venv/bin/python scripts/summarize.py` → the lr 1e-3 table in `results/summary.md`.
5. `scripts/run_lr.sh` (~40 min) → `results/train-softmax-seed0-lr0.003.json` (≈ 0.918) and `results/train-cvqiaa-seed0-lr0.003.json` (≈ 1.042).

Version 2:

6. `.venv/bin/python src/paper_v2_blueprint.py` and `.venv/bin/python src/paper_v2_replacement.py` → output of the authors' tests (section 6).
7. `.venv/bin/python scripts/bench.py` (~1 min) → the section 2 table in `results/bench.json`. Memory will match exactly, time approximately.
8. `scripts/run_v2.sh` (~50 min) and `scripts/run_v2_ablations.sh` (~30 min), then `.venv/bin/python scripts/summarize_v2.py` → the section 1 table in `results/summary-v2.md` together with the learned σ₀.

Version 3:

9. `.venv/bin/python scripts/check_v3.py` (a few seconds) → `results/v3-checks.json`: match with v2, the section 1.1 table, the number of exact zeros and the phase-rotation check.

## Layout

- `src/models.py`: the shared model (the paper's WaveLLM skeleton) and attention variants: softmax, born, none, cvqiaa (v1), cvqiaa2 (v2) and its control variants.
- `scripts/train.py`: training one variant: character tokens, AdamW, 100 warm-up steps and cosine, gradient clipping 1.0. Validation uses the same 50 batches for all runs.
- `results/v2-verification.json`: findings of the independent reviewers on version 2 and the attempts to refute them.

The paper's code is distributed under CC0 1.0 (see `paper/README TXT.txt`).
