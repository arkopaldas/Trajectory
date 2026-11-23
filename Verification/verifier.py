"""The verification itself: decides, cell by cell, whether the closed loop (trained network + true dynamics) can violate a
safety constraint, and returns SAFE / UNSAFE / UNKNOWN.

How it works
  1. The start region (and, for the unicycle, the heading range) is split into cells (config.START_CELLS_*).
  2. For each cell and each step k = 0..T a Marabou query asks: is there a start state in this cell whose closed-loop rollout
     violates an active constraint at step k? (job_list / job_property)
  3. UNSAT for all queries of all cells = SAFE. A SAT answer is replayed on the true dynamics (utils.exact_rollout); only a
     replayed violation counts as a counterexample (UNSAFE). Anything else (timeout, SAT that does not replay) = UNKNOWN.
Counterexamples are appended to Datasets/Counterexamples and a per-cell table is saved to Verification/Results.
"""
import time
import numpy as np

import config
import constraints
import utils
from marabou_encoder import Encoding, build_rollout, get_spec


# Fails early with an install hint if Maraboupy is missing.
def check_marabou():
    try:
        import maraboupy   # noqa: F401
    except ImportError as err:
        raise ImportError("Maraboupy is not installed. Install it with: pip install maraboupy onnx onnxruntime") from err


# --------------------------------------------------------------------------- the queries of one start cell
def job_list(env):
    """
    Closed-loop queries for one start cell, shallow to deep. A query of step k asks Marabou for a start state in the cell
    whose closed-loop rollout violates an active constraint at step k (the state s_k, or the control u_k).
    GOAL_MODE "final" adds the goal check to step T; "any" adds one extra query: no state s_0..s_T is in the goal.
    """
    jobs = []
    # One query per step k = 0..T (only those with something to check).
    for k in range(env.T + 1):
        has_u = k < env.T and "control" in constraints.active_constraints(env)
        has_goal = k == env.T and config.GOAL_MODE == "final"
        if constraints.state_violation_atoms(env, 0.0) or has_u or has_goal:
            jobs.append((k, "step"))
    if config.GOAL_MODE == "any":
        jobs.append((env.T, "goal_any"))
    return jobs


def job_property(env, states, controls, k, kind):
    """The violation to look for, as a conjunction of disjunctions of bound cases on the query variables."""
    eps = config.PROPERTY_EPS

    # Maps atom coordinates (0 = x, 1 = y, 2 = theta / control index) to the Marabou variables of the state or control vector.
    def on(atoms, variables):
        return [[(int(variables[c]), lo, hi) for c, lo, hi in case] for case in atoms]

    if kind == "goal_any":
        return [on(constraints.goal_violation_atoms(env, eps), states[t]) for t in range(env.T + 1)]
    cases = on(constraints.state_violation_atoms(env, eps), states[k])
    control_atoms = constraints.control_violation_atoms(env, eps) if k < env.T else []
    if control_atoms:
        cases += on(control_atoms, controls[k])
    if k == env.T and config.GOAL_MODE == "final":
        cases += on(constraints.goal_violation_atoms(env, eps), states[k])
    return [cases] if cases else []


def check_job(env, layers, spec, session, cell, k, kind, deadline):
    """
    Returns (status, counterexample, stats). status: 'unsat' (no violation in this cell at this step), 'sat' (a replayed
    counterexample), or 'unknown' (timeout, or a Marabou SAT that does not replay on the true dynamics even after refining).
    """
    stats = {"queries": 0, "by_bounds": 0, "spurious": 0, "build_s": 0.0, "solve_s": 0.0}
    # Encoding refinement levels: the unicycle is retried with finer piecewise-linear bands if a SAT does not replay.
    levels = range(config.MAX_REFINEMENTS + 1) if env.model == "unicycle" else [0]   # the Simple model is encoded exactly
    for level in levels:
        t0 = time.time()
        # Number of policy copies in the query: controls u_0..u_(k-1) drive the dynamics up to s_k; u_k is added only if the
        # property also concerns the control at step k.
        n_controls = k + 1 if (k < env.T and kind == "step" and "control" in constraints.active_constraints(env)) else k
        q, states, controls = build_rollout(env, layers, spec, cell, k, n_controls, Encoding(level))
        for cases in job_property(env, states, controls, k, kind):
            q.add_property(cases, config.USE_BOUND_PRUNING)
        stats["build_s"] += time.time() - t0
        # The propagated bounds already prove that no violation exists: no Marabou call needed.
        if q.infeasible and config.USE_BOUND_PRUNING:
            stats["by_bounds"] += 1
            return "unsat", None, stats
        remaining = deadline - time.time()
        if remaining <= 0:
            return "unknown", None, stats
        t1 = time.time()
        status, values = q.solve(min(config.QUERY_TIMEOUT, remaining))
        stats["queries"] += 1
        stats["solve_s"] += time.time() - t1
        if status != "sat":
            return ("unsat" if status == "unsat" else "unknown"), None, stats
        # Marabou found a start state. Trust it only if the true closed-loop rollout really violates a constraint.
        # Marabou found a start state. Trust it only if the exact closed-loop rollout (true dynamics, onnxruntime) really violates a
        # constraint; the bands around cos / sin / products could have allowed a spurious solution.
        x0 = np.clip(np.array([values[int(v)] for v in q.start_vars]), cell[0], cell[1])
        rolled_states, rolled_controls = utils.exact_rollout(env, session, x0)
        violations = constraints.trajectory_violation(env, rolled_states, rolled_controls)
        if constraints.is_unsafe(violations):
            ce = {"start": x0, "states": rolled_states, "controls": rolled_controls, "violations": violations,
                  "step": k, "kind": kind, "build_ms": stats["build_s"] * 1000, "solve_ms": stats["solve_s"] * 1000}
            return "sat", ce, stats
        stats["spurious"] += 1       # the band around cos / sin / products allowed this one: refine and ask again
    return "unknown", None, stats


# --------------------------------------------------------------------------- the whole verification
def run_verification(model, onnx_path=None, max_counterexamples=1, save=True):
    """
    Verifies one network by closed-loop Marabou queries, cell by cell.
      UNSAFE  : a start state whose closed-loop rollout violates an active constraint was found (and replayed)
      SAFE    : every Marabou query of every cell is UNSAT
      UNKNOWN : no counterexample, but a query timed out / was not conclusive / time ran out
    """
    check_marabou()
    env = utils.Environment(model)
    path = onnx_path or utils.network_file(model)
    layers, session, spec = utils.load_onnx_layers(path), utils.make_session(path), get_spec(model)
    cells, jobs = utils.make_cells(env), job_list(env)
    deadline = time.time() + config.TOTAL_TIMEOUT
    started = time.time()

    print(f"\n[{config.MODELS[model]['name']} verification] network: {path}")
    print(f"[{config.MODELS[model]['name']} verification] active constraints: {', '.join(constraints.active_constraints(env))} "
          f"(goal mode: {config.GOAL_MODE}) | {len(cells)} start cells x {len(jobs)} queries | horizon T = {env.T}")

    # ---- Main loop over the start cells; stops early once max_counterexamples counterexamples have been collected ----
    rows, counterexamples, unknown, processed = [], [], False, 0
    for index, cell in enumerate(cells):
        if time.time() > deadline:
            unknown = True
            break
        cell_status, totals, found = "SAFE", {"queries": 0, "by_bounds": 0, "spurious": 0, "build_s": 0.0, "solve_s": 0.0}, None
        for k, kind in jobs:
            status, ce, stats = check_job(env, layers, spec, session, cell, k, kind, deadline)
            for key in totals:
                totals[key] += stats[key]
            if status == "sat":
                cell_status, found = "UNSAFE", ce
                counterexamples.append(ce)
                break
            if status != "unsat":
                cell_status, unknown = "UNKNOWN", True
        processed += 1
        rows.append({
            "CELL": index, "LOWER": np.round(cell[0], 6).tolist(), "UPPER": np.round(cell[1], 6).tolist(), "STATUS": cell_status,
            "MARABOU QUERIES": totals["queries"], "DECIDED BY BOUNDS": totals["by_bounds"], "SPURIOUS SAT": totals["spurious"],
            "BUILD TIME (s)": round(totals["build_s"], 2), "SOLVE TIME (s)": round(totals["solve_s"], 2),
            "COUNTEREXAMPLE START": np.round(found["start"], 6).tolist() if found else "",
            "VIOLATIONS": {k: round(v, 4) for k, v in found["violations"].items()} if found else "",
        })
        print(f"  cell {index + 1}/{len(cells)} {cell_status:<8} queries={totals['queries']} by-bounds={totals['by_bounds']} "
              f"build={utils.format_seconds(totals['build_s'])} solve={utils.format_seconds(totals['solve_s'])}"
              + (f" | start={np.round(found['start'], 4).tolist()} violations={rows[-1]['VIOLATIONS']}" if found else ""))
        if len(counterexamples) >= max_counterexamples:
            break

    # Final verdict: UNSAFE if a replayed counterexample exists, UNKNOWN if something was inconclusive or not processed, else SAFE.
    if counterexamples:
        verdict = "UNSAFE"
    elif unknown or processed < len(cells):
        verdict = "UNKNOWN"
    else:
        verdict = "SAFE"
    if save:
        utils.save_counterexamples(env, counterexamples)
        utils.save_report(model, rows, "Verification")
    print(f"[{config.MODELS[model]['name']} verification] {verdict} | {len(counterexamples)} counterexample(s) | "
          f"time {utils.format_seconds(time.time() - started)}")
    return {"verdict": verdict, "counterexamples": counterexamples, "rows": rows, "env": env}
