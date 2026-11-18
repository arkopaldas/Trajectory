import numpy as np

import config
import constraints
import utils
from marabou_encoder import Encoding, build_rollout, get_spec

# Marabou-free consistency test. It checks, on random start states, that
#   1. the Linear/ReLU stack read from the ONNX file reproduces onnxruntime,
#   2. the constraint system that would be sent to Marabou contains the TRUE closed-loop trajectory
#      (every propagated bound, every cos / sin / product band and every angle wrap holds on the exact values),
#   3. the property atoms agree with the violation degrees used by the replay and by the CEGIS loss.
# If this passes, a Marabou SAT / UNSAT is about the right problem. Marabou itself is not needed.

SAMPLES = 20
DEPTHS = (1, 3, 8)


def test_model(model, rng):
    env = utils.Environment(model)
    path = utils.network_file(model)
    layers, session, spec = utils.load_onnx_layers(path), utils.make_session(path), get_spec(model)
    cells = utils.make_cells(env)
    problems = 0
    tested = 0

    probe = rng.uniform(-5, 100, size=(256, env.state_dim)).astype(np.float32)
    err = np.max(np.abs(utils.forward(layers, probe) - session.run(["output"], {"input": probe})[0]))
    print(f"[{model}] max |extracted layers - onnxruntime| = {err:.2e}")
    problems += err > 1e-3

    for cell in [cells[0], cells[len(cells) // 2], cells[-1]]:
        for depth in DEPTHS:
            # if depth > env.T:
            #     continue
            n_controls = depth + 1
            q, states, controls = build_rollout(env, layers, spec, cell, depth, n_controls, Encoding(0))
            for _ in range(SAMPLES):
                x0 = rng.uniform(cell[0], cell[1])
                rolled_states, rolled_controls = utils.exact_rollout(env, session, x0)
                used = rolled_controls[:depth]       # the encoding assumes the applied controls are inside their bounds
                if np.any(used < np.array(env.ctrl_lo) - config.PROPERTY_EPS) or np.any(used > np.array(env.ctrl_hi) + config.PROPERTY_EPS):
                    continue
                tested += 1
                val = q.evaluate(x0)
                got = np.array([val[states[t]] for t in range(depth + 1)])
                gap = np.max(np.abs(got - rolled_states[:depth + 1]))
                bad = q.violations_of(val, tol=1e-4)
                if gap > 5e-3 or bad:
                    problems += 1
                    print(f"  FAIL cell {cell[0]} depth {depth}: state gap {gap:.2e} {bad[:2]}")
            print(f"[{model}] cell lower={np.round(cell[0], 3).tolist()} depth {depth}: {q.n} variables, "
                  f"{sum(len(o[2]) for o in q.ops if o[0] == 'relu')} ReLUs, {len(q.ops)} constraint blocks")

    print(f"[{model}] {tested} sampled trajectories satisfied the control-bound assumption and were checked")
    problems += tested == 0

    # property atoms versus violation degrees
    eps = config.PROPERTY_EPS
    for _ in range(2000):
        s = rng.uniform(-5, 105, size=env.state_dim)
        in_atoms = any(all(lo <= s[c] <= hi for c, lo, hi in case) for case in constraints.state_violation_atoms(env, eps))
        d = []
        active = constraints.active_constraints(env)
        if "obstacle" in active:
            d.append(max(constraints.obstacle_depths(env, s[0], s[1])))
        if "workspace" in active:
            d.append(constraints.workspace_degree(env, s[0], s[1]))
        if "theta" in active:
            d.append(constraints.theta_degree(env, s[2]))
        if in_atoms != (max(d) >= eps):
            problems += 1
            print("  FAIL: atoms and degrees disagree at", s)
            break
    return problems


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    total = sum(test_model(m, rng) for m in ("simple", "unicycle"))
    print("\nSELFTEST PASSED" if total == 0 else f"\nSELFTEST FAILED ({total} problem(s))")
