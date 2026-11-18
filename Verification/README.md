# Verification

Closed-loop formal verification (Maraboupy) and CEGIS fine-tuning of the trained Simple / Unicycle policies.
No script takes arguments: edit `config.py`, then run the script.

| File | Purpose |
|---|---|
| `config.py` | network / dataset size, **constraint flags**, tolerances, start-cell grid, Marabou limits, CEGIS settings |
| `constraints.py` | the safety constraints, defined once (Marabou property, exact replay, CEGIS loss all use it) |
| `utils.py` | environment from the dataset CSV, ONNX reader, exact rollout, cells, counterexample CSV |
| `marabou_encoder.py` | closed-loop Marabou encoding (policy copies chained through the dynamics) |
| `verifier.py` | cell-by-cell queries, replay of every Marabou SAT, SAFE / UNSAFE / UNKNOWN |
| `cegis.py` | `L_total = L_ori + LAMBDA_CE * L_ce`, fine-tuning, the CEGIS loop |
| `verify_simple.py`, `verify_unicycle.py` | verification entry points |
| `cegis_simple.py`, `cegis_unicycle.py` | CEGIS entry points |
| `selftest.py` | Marabou-free check that the encoding contains the true trajectories |

## How verification works
For every start cell (x, y [, theta]) and every step k = 0..T, Marabou receives ONE closed-loop query: k chained copies of the
policy, `s_(i+1) = f(s_i, pi(s_i))`, and the property "an active constraint is violated at step k". SAT gives a start state;
it is replayed with onnxruntime and the true dynamics, and only a replayed violation is reported (UNSAFE).
All queries UNSAT -> SAFE. Timeout / inconclusive -> UNKNOWN.

* Simple: dynamics are linear, so the encoding is exact.
* Unicycle: Marabou has no cos / sin / product. They are enclosed by piecewise-linear bands (ReLU interpolants with a proven
  error bound of about `TRIG_STEP^2 / 8`); the wrap of theta is an exact disjunction. The true value is always inside the band, so
  UNSAT remains a proof; a SAT that does not replay is solved again with half the spacing (`MAX_REFINEMENTS`).
* Controls applied before step k are assumed in bounds (up to `PROPERTY_EPS`) in the step-k query; an earlier break is caught by the
  earlier query, so no unsafe trajectory is lost.
* `SAFE` means: no violation of degree >= `PROPERTY_EPS` (a floating point solver cannot resolve smaller ones).

## Outputs
* `Results/<N> Trained/<Model> Model Verification.csv`: one row per cell.
* `../Datasets/Counterexamples/<N> Trajectories/<Model> Model Counterexamples.csv`: dataset schema, STATUS = COUNTEREXAMPLE.
* `../Networks/<N> Trained/<Model>_CEGIS.onnx`: fine-tuned network (the original is never changed).

## Requirements
Python 3.11, `maraboupy`, `onnx`, `onnxruntime`, `numpy`, `pandas`; CEGIS also `torch` and the dependencies of `Neural Networks`.
