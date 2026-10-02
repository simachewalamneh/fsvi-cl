# fsvi-cl

Function-space variational inference for continual learning (S-FSVI, Rudner et al., ICML 2022), in JAX.
Goal of this repo: (1) a clean S-FSVI baseline, (2) honest comparisons, (3) then combine it with
predictive-coding (PC) approaches **one at a time**.

## Status: Stage 1 scaffold (not yet validated against the paper)

Implemented: Bayesian MLP (mean-field Gaussian, multi-head), and three methods on the same loop:

| method     | regulariser for tasks >= 1                                              |
|------------|-------------------------------------------------------------------------|
| `finetune` | none (deterministic, forgetting lower bound)                            |
| `vcl`      | weight-space KL(q_t \|\| q_{t-1}), no coreset                           |
| `fsvi`     | KL over *function outputs* at stored context points of earlier tasks    |

Run: `python -m fsvi_cl.cli --method fsvi --dataset synthetic` (or `--dataset split_mnist`, needs keras + internet).
Tests: `python -m pytest -q`.

## Known simplifications / not yet verified
- Function-space Gaussians are diagonal and estimated from weight samples (not the paper's exact approximation).
- Task 0 uses a weight-space KL to N(0, 1) (weight `beta_first`); context points are random subsets.
- Multi-head only; evaluation uses posterior-mean weights.
- Only run on a synthetic stream so far, where all methods are within noise (fine-tune barely forgets in
  multi-head). **Split-MNIST has not been run yet and no paper numbers have been reproduced.**

## Roadmap (one step at a time, each step compared on the same benchmark and seeds)
1. **Stage 1 (this commit):** baselines + S-FSVI scaffold, tests.
2. **Stage 2 - validate:** real Split-MNIST (multi-head and single-head), >=5 seeds, results table script;
   compare against published S-FSVI / VCL numbers; fix any gap before going further.
3. **Stage 3 - predictive coding, one by one:** each PC approach is its own module + config + ablation, e.g.
   3a. PC inference (FabricPC) for the likelihood term in place of backprop, keeping the fSVI regulariser;
   later items chosen with the mentor. Nothing is merged unless it beats or explains itself against Stage 2.
