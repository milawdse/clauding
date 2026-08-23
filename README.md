# Choice-Theory Personality Puzzles — V0

A questionnaire grounded in William Glasser's *Choice Theory*, a bank of
dilemma puzzles, and a model that **predicts which solution you will pick before
you pick it** — then shows you the reasoning that produced the guess.

**V0 uses no language model at all.** Everything here is numpy, scipy and
arithmetic. It runs on a laptop CPU in seconds, and it exists to answer one
question before any GPU is bought:

> Does personality conditioning actually predict a person's choice, or is it
> decoration?

## The answer, on synthetic data

```
  uniform random                     acc  25.1%   ECE 0.001
  modal option (no profile)          acc  26.3%   ECE 0.017
  contest model, SHUFFLED profile    acc  25.1%   ECE 0.314
  oracle (true latent + true params) acc  63.1%   ECE 0.086
  contest model, real profile (V0)   acc  56.3%   ECE 0.009

  fitted beta = 2.394   95% CI [2.296, 2.553]
```

Read those rows in order — the interesting ones are the middle two.

* Handed **somebody else's profile**, the model drops to 25.1%, which is exactly
  chance on four options. The personality signal is doing *all* of the work; none
  of the accuracy comes from option-ordering artefacts or base rates.
* The **oracle** — true latent needs, true generating parameters, no
  questionnaire noise — reaches 63.1%. That is the ceiling. V0 gets 56.3%, so
  there are **6.8 points of headroom in total**, and most of that gap is
  questionnaire measurement error rather than anything a bigger model would fix.

That second point is the main finding. Before building V0 it was reasonable to
assume a 32B model would predict much better than arithmetic. The ceiling says it
cannot: there is very little room above the linear model, and the cheapest way to
buy accuracy is better questionnaire items, not a larger predictor.

## Try it

```bash
pip install numpy scipy pyyaml

python -m glasser_puzzles.cli demo          # synthetic person, no typing
python -m glasser_puzzles.cli play          # answer it yourself
python -m glasser_puzzles.cli verify-log    # audit the sealed predictions
```

Evaluation and validation:

```bash
python -m glasser_puzzles.eval.recovery --n 2000     # does the quiz measure anything?
python -m glasser_puzzles.eval.prediction --n 1200   # gates 0 and 1
python -m glasser_puzzles.puzzles.validate --strict  # vet the puzzle bank
python -m pytest tests/ -q
```

## How it works

**23 questions.** Five basic needs (Survival, Love & Belonging, Power, Freedom,
Fun) measured by four items each, balanced two forward and two reverse, plus
three items on internal versus external control psychology.

**Scoring is ipsative.** Raw Likert sums measure how agreeable you are, not what
you need: someone who agrees with everything scores high on all five. Since
Glasser's model is about *relative* need strength anyway, each person's own mean
is subtracted first. A respondent who answers 5 to all 23 items comes out
perfectly flat, which is the correct answer — they told us nothing.

**Puzzles are profile-blind.** 30 hand-authored dilemmas, three for each of the
ten need-contrast pairs, each with four defensible solutions. What adapts to you
is *which contrast gets served next*, chosen by how uncertain the profile
currently is and which contrasts are under-covered. The puzzles themselves never
see your profile — a generator that did would write the profile-matching option
to be the most attractive one, and the predictor would then be predicting its own
bias rather than you.

**The model.** For an option serving need `n`:

```
P(choose i) = softmax( beta * s[n_i] + alpha[n_i] )
```

`beta` is how strongly relative need strength drives choice. `alpha` absorbs the
fact that some options are more appealing regardless of who is answering —
without it, `beta` would soak up the base rate and look predictive while only
reproducing what everyone picks.

Logistic rather than gradient-boosted **on purpose**: the reveal has to explain
itself, and a black box cannot. Interpretability is a hard requirement here, not
a preference.

**Predictions are sealed.** The guess and its reasoning are hashed and written to
an append-only log *before* the scenario is displayed. `verify-log` re-hashes
every entry and checks each was sealed before its answer was recorded. A
reasoning trace written after the answer is known is rationalisation, not
prediction — and it is otherwise undetectable from outside, so the log is the
evidence.

**Your profile sharpens as you play.** Each answer is a Laplace update on the
contested needs. Per-need questionnaire confidence sets the prior width, so a
need whose four items disagreed with each other moves quickly under evidence,
while one measured cleanly holds its ground.

## Limitations — please read these

**There is no real human data here, and none of the numbers above involve
people.** Synthetic personas can establish internal consistency and
non-triviality. They cannot establish that this predicts *humans*. It has been
shown to predict a simulator, and that is all.

**The simulator is deliberately mis-specified** — per-persona intercept jitter,
extra logit noise, and 10% near-random responders, none of which the predictor's
functional form can represent. `eval/prediction.py` **refuses to report a
number** if the simulator has no unmodelled structure, because a correctly
specified simulator would score near 100% and mean nothing. That guard is
enforced in code, not in a comment.

**The puzzle balance check reads need composition, not writing.** It catches
option sets where one need dominates for everybody. It cannot tell whether one
option is simply worded more attractively. That needs a human reader.

**These items are original work** written from the constructs in Glasser's
published writing. They are deliberately *not* a reproduction of the Glasser
Institute's "Basic Needs Profile", which is a proprietary instrument. No validity
claims are made for them, and none should be inferred from the recovery numbers —
those show the scorer can invert its own simulator, which is a much weaker claim
than construct validity.

**This is not a psychometric instrument and not a diagnostic tool.** It is a
self-reflection and research prototype. Session data stays on your machine.

## What would actually improve it

In descending order of expected value:

1. **Thirty to fifty real sessions.** The prediction loop already logs
   `(profile, puzzle, prediction, actual)` as a side effect of normal use, so
   real labelled data arrives free. Refit `beta` and `alpha` on it — that turns
   the shipped parameters from an assumption into a measurement.
2. **Better questionnaire items.** The oracle gap says measurement error, not
   model capacity, is the binding constraint. Items with higher discrimination
   are worth more than a bigger model.
3. **More puzzles per contrast**, so the selector has room to avoid repeats in a
   longer session.
4. Only then, a language model — and it has under seven points to win.

## Layout

```
glasser_puzzles/
  needs/       item bank, constructs, deterministic scoring
  profile/     heuristic vocabulary and rule-based profile cards
  puzzles/     schema, contrast pairs, the frozen bank, selector, validator
  predict/     contest model, template explainer, sealing
  update/      Bayesian belief update
  synth/       personas, IRT response model, mis-specified answer simulator
  eval/        instrument recovery, prediction lift and the gates
  cli.py       play / demo / verify-log
```
