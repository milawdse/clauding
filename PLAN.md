# Study Plan: Reasoning Capacity in Qwen2.5 (0.5B/3B/7B) via LoRA on `color_cube_rotation`

Hardware target: single RTX 4090 (24GB VRAM).

## Research questions

1. How does parameter count (0.5B / 3B / 7B) affect a Qwen2.5 model's ability to
   acquire the cube-rotation algorithm via LoRA SFT?
2. Does supervising on algorithmically-generated **intermediate reasoning steps**
   (a scratchpad of the cube state after each rotation) improve accuracy and
   generalization vs. training on final answers only — and does the benefit
   differ by model size?
3. At what LoRA capacity (rank) does each model size saturate on this task, and
   do the trained models generalize to longer rotation chains than seen in
   training (evidence of a learned algorithm) or plateau/collapse (evidence of
   pattern matching)?

## 1. Background reading (~1 day)

- Qwen2.5 Technical Report — architecture per size (layers/hidden/heads/GQA),
  tokenizer, context length, chat template, license per checkpoint size.
- Hu et al., *LoRA: Low-Rank Adaptation of LLMs* (2021); Dettmers et al.,
  *QLoRA* (2023) — rank/alpha/target-module and 4-bit training tradeoffs.
- reasoning-gym paper (NeurIPS 2025 Spotlight) and repo — task/scoring design,
  RLVR framing (https://github.com/open-thought/reasoning-gym).
- Nye et al., *Show Your Work: Scratchpads for Intermediate Computation*
  (2021); Wei et al., CoT prompting; Lightman et al., process supervision —
  motivates why algorithmic step traces should help small models most.

## 2. The task: `color_cube_rotation`

Source: `reasoning_gym/cognition/color_cube_rotation.py`.

- Config: `min_rotations=1`, `max_rotations=3`, `seed`, `size=500` (defaults).
- A cube has 6 colored faces. A random sequence of rotations is applied
  ("rotate so the side that was at X is now at top"). The question asks for
  the color of one named face after all rotations; scorer does an exact
  lowercased/stripped string match (1.0 correct, 0.01 wrong, 0.0 for `None`
  — vestigial RLVR-style partial credit, irrelevant for SFT).
- **Problem:** the shipped dataset only exposes the final Q/A pair, not a
  step-by-step trace. We need to generate the middle steps ourselves.

### Data generation algorithm (the "middle-step reasoning" generator)

Don't reimplement cube rotation from scratch — **import reasoning-gym's own
`Cube`/`Side` classes and call the same rotation methods** it uses internally
(e.g. `rotate_front_to_top()`), so our generated trace is guaranteed
consistent with their answer key. For each generated example:

1. Print the initial state (6 face→color assignments).
2. For each rotation in the sequence: state which face moves where, apply it
   via the library's own method, then print the resulting full 6-face state
   as a compact table (this is the "middle-step reasoning").
3. End with the answer line in the exact format their scorer expects
   ("Provide only the color as your final answer." → bare color word).

Build, from the same underlying scrambles:
- **CoT variant**: assistant target = full step trace + final answer.
- **No-CoT variant**: assistant target = final answer only (ablation control
  for research question 2 — same questions/answers, only the intermediate
  supervision differs).
- **PoT (Program of Thoughts) variant**: assistant target = a short Python
  program against a fixed `Cube(top=..., ..., bottom=...)` /
  `.rotate_to_top(side)` API (see `data_gen/pot_library.py`), ending in a
  `print(...)` of the requested face. No natural-language reasoning at all —
  the model only translates the story into API calls; an interpreter
  (`data_gen/pot_executor.py`, sandboxed subprocess, 5s timeout) executes the
  program and the printed output is graded, using the same
  1.0/0.01/0.0 convention as reasoning-gym's own `score_answer`. Built from
  the *same* seeds as the CoT/no-CoT splits, so all three are directly
  comparable per-example. This is a fourth arm for research question 2: does
  offloading computation to code (vs. NL scratchpad, vs. no scratchpad) shift
  where each model size saturates?

Splits (paired across CoT/no-CoT):
- `train`: min=1, max=3 rotations, ~8k examples.
- `val`: same range, disjoint seed, ~1k.
- `test_seen`: same range, disjoint seed, ~1k.
- `test_extrapolate`: min=4, max=6 rotations (never seen in training), ~1k —
  tests compositional generalization vs. memorization.

Format examples with Qwen2.5's chat template (system/user/assistant turns).

## 3. Models & LoRA configs

| Model | Params | Base precision | LoRA r | alpha | Target modules | Est. VRAM |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B(-Instruct) | ~0.5B | bf16 | 8–16 | 16–32 | q/k/v/o + gate/up/down proj | ~3–5 GB |
| Qwen2.5-3B(-Instruct) | ~3B | bf16 | 16 | 32 | same | ~10–14 GB |
| Qwen2.5-7B(-Instruct) | ~7B | **4-bit NF4 (QLoRA)** | 16–32 | 32–64 | same | ~10–14 GB w/ grad ckpt |

Notes:
- Include MLP projections (`gate/up/down_proj`), not just attention — this
  task needs "state update" computation that LoRA on attention-only often
  under-fits.
- Verify each checkpoint's exact license on its HF model card before any use
  beyond personal research (sizes differ: some Apache-2.0, some under the
  Qwen license).
- Additionally run a **LoRA rank sweep** on the 3B model (r ∈ {4,8,16,32,64})
  to directly probe research question 3 (minimal capacity to fit the task).

## 4. Training setup

- Stack: `transformers` + `peft` (+`trl` `SFTTrainer` optional) + `bitsandbytes`
  (7B only) + `accelerate`.
- Sequence length: traces are short (few hundred tokens) → `max_seq_len` ~1024
  is safe, allows large batch.
- Effective batch size ~64–128 via micro-batch + gradient accumulation.
- LR 1e-4–2e-4, cosine schedule, ~3–5% warmup, bf16 compute.
- 2–3 epochs over 8k train examples; track val accuracy each epoch, stop
  early — this is an algorithmic task, easy to overfit past saturation.

## 5. Evaluation & analysis

Primary: exact-match accuracy on the final color (identical scorer to
reasoning-gym).

Secondary (the actual "capacity vs. reasoning" analysis):
- **Accuracy vs. chain length**: break down accuracy by rotation count
  (1–3 seen, 4–6 extrapolation), plotted per model size × {CoT, no-CoT}. This
  is the core capacity/depth curve.
- **Step-level faithfulness**: for CoT models, parse the model's stated
  intermediate cube state after each rotation and diff it against the
  ground-truth simulator step-by-step (not just the final answer). Report
  per-step accuracy and the distribution of "first wrong step" — this
  distinguishes models that execute the algorithm from ones that produce
  plausible-looking but ungrounded traces while guessing the final color.
- **CoT vs. no-CoT** at matched data/compute per model size → does
  intermediate supervision help 0.5B proportionally more than 7B?
- **Rank sweep** on 3B → accuracy-vs-rank curve, find saturation point.
- *(Optional stretch)* linear probe on the residual stream after each
  rotation step to predict the true face-color state, to check whether
  correct "belief" is linearly decodable even when the final emitted answer
  is wrong (computation error vs. read-out error).

## 6. Compute budget (single RTX 4090)

- Data generation: pure Python, no GPU, minutes for thousands of examples.
- 0.5B LoRA SFT run: ~10–20 min.
- 3B LoRA SFT run: ~30–60 min.
- 7B QLoRA SFT run: ~1–2 hr (4-bit, grad checkpointing, smaller micro-batch).
- Rank sweep (5 configs, 3B): ~3–5 hr.
- Full matrix (3 sizes × {CoT, no-CoT} = 6 runs) + rank sweep + extrapolation
  eval: roughly 6–10 GPU-hours total — a single long session or a weekend.

## 7. Deliverables

- Data-gen script wrapping reasoning-gym's `Cube`/`Side` classes, emitting
  paired CoT/no-CoT SFT datasets with seen/extrapolation splits.
- One parametrized training script (or per-size config files) covering all
  three model sizes and the rank sweep.
- Eval script producing overall + per-rotation-count + per-step metrics and
  the accuracy-vs-length / accuracy-vs-rank plots.
- Short written report answering the three research questions with the
  resulting plots and step-faithfulness analysis.
