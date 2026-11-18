import numpy as np

import config

# Every safety constraint is defined ONCE, here, through a "violation degree":
#     degree >= PROPERTY_EPS  ->  the constraint is violated      (degree > 0 means "unsafe side")
# The same definition is used by
#   * the Marabou queries         (violation_atoms: the degree >= eps region written as variable bounds),
#   * the exact replay            (trajectory_violation),
#   * the CEGIS loss              (ce_penalties: a real number saying how much the trajectory is violating).
# Constraints that are switched off in config.py are never reported, encoded or penalised.

INF = float("inf")


def active_constraints(env):
    if config.GOAL_MODE not in ("final", "any", "off"):
        raise ValueError("config.GOAL_MODE must be 'final', 'any' or 'off'")
    names = []
    if config.CHECK_CONTROL_BOUNDS:
        names.append("control")
    if config.CHECK_OBSTACLES:
        names.append("obstacle")
    if config.CHECK_WORKSPACE:
        names.append("workspace")
    if config.CHECK_THETA_BOUNDS and env.theta_bounds is not None:
        names.append("theta")
    if config.GOAL_MODE != "off":
        names.append("goal")
    return names


# --------------------------------------------------------------------------- violation degrees
# These work on numpy arrays and on torch tensors (xp = numpy or torch).
def _relu(xp, t):
    return xp.clamp(t, min=0.0) if xp.__name__ == "torch" else np.maximum(t, 0.0)


def _tsum(xp, t):
    """Sum over the time axis (axis 1) of a (B, T+1) or (B, T) array / tensor."""
    return t.sum(dim=1) if xp.__name__ == "torch" else t.sum(axis=1)


def _tmin(xp, t):
    return t.min(dim=1).values if xp.__name__ == "torch" else t.min(axis=1)


def obstacle_depths(env, x, y, xp=np):
    """One value per obstacle: how deep (L-infinity) the point is inside the obstacle inflated by the margin. > 0 = inside."""
    m = env.margin
    depths = []
    for x1, y1, x2, y2 in env.obstacles:
        dx = xp.minimum(x - (x1 - m), (x2 + m) - x)
        dy = xp.minimum(y - (y1 - m), (y2 + m) - y)
        depths.append(xp.minimum(dx, dy))
    return depths


def workspace_degree(env, x, y, xp=np):
    xmin, ymin, xmax, ymax = env.workspace
    return xp.maximum(xp.maximum(xmin - x, x - xmax), xp.maximum(ymin - y, y - ymax))


def goal_degree(env, x, y, xp=np):
    gx1, gy1, gx2, gy2 = env.goal
    return xp.maximum(xp.maximum(gx1 - x, x - gx2), xp.maximum(gy1 - y, y - gy2))


def theta_degree(env, theta, xp=np):
    return xp.maximum(env.theta_bounds[0] - theta, theta - env.theta_bounds[1])


def control_degree(env, u, xp=np):
    degree = None
    for j in range(len(env.ctrl_lo)):
        d = xp.maximum(env.ctrl_lo[j] - u[..., j], u[..., j] - env.ctrl_hi[j])
        degree = d if degree is None else xp.maximum(degree, d)
    return degree


# --------------------------------------------------------------------------- exact replay (numpy)
def trajectory_violation(env, states, controls):
    """
    states (T+1, d), controls (T, c) of one closed-loop rollout.
    Returns {active constraint: worst violation degree over the trajectory}. Degree > 0 = violated.
    The goal degree is the distance of the final state ("final") or of the closest state ("any") to the goal.
    """
    active = active_constraints(env)
    x, y = states[:, 0], states[:, 1]
    out = {}
    if "control" in active:
        out["control"] = float(np.max(control_degree(env, controls)))
    if "obstacle" in active:
        out["obstacle"] = float(max(np.max(d) for d in obstacle_depths(env, x, y)))
    if "workspace" in active:
        out["workspace"] = float(np.max(workspace_degree(env, x, y)))
    if "theta" in active:
        out["theta"] = float(np.max(theta_degree(env, states[:, 2])))
    if "goal" in active:
        d = goal_degree(env, x, y)
        out["goal"] = float(d[-1] if config.GOAL_MODE == "final" else np.min(d))
    return out


def is_unsafe(violations, eps=None):
    eps = config.CONFIRM_EPS if eps is None else eps
    return any(v >= eps for v in violations.values())


# --------------------------------------------------------------------------- Marabou property atoms
# An "atom set" is a list of cases (a disjunction). A case is a list of (coordinate, lower, upper) bounds that hold together.
def state_violation_atoms(env, eps):
    """Cases in which ONE state (x, y[, theta]) violates an active state constraint."""
    active = active_constraints(env)
    cases = []
    if "obstacle" in active:
        m = env.margin
        for x1, y1, x2, y2 in env.obstacles:
            cases.append([(0, x1 - m + eps, x2 + m - eps), (1, y1 - m + eps, y2 + m - eps)])
    if "workspace" in active:
        xmin, ymin, xmax, ymax = env.workspace
        cases += [[(0, -INF, xmin - eps)], [(0, xmax + eps, INF)], [(1, -INF, ymin - eps)], [(1, ymax + eps, INF)]]
    if "theta" in active:
        cases += [[(2, -INF, env.theta_bounds[0] - eps)], [(2, env.theta_bounds[1] + eps, INF)]]
    return cases


def control_violation_atoms(env, eps):
    """Cases in which ONE control vector violates its bounds."""
    if "control" not in active_constraints(env):
        return []
    cases = []
    for j in range(len(env.ctrl_lo)):
        cases += [[(j, -INF, env.ctrl_lo[j] - eps)], [(j, env.ctrl_hi[j] + eps, INF)]]
    return cases


def goal_violation_atoms(env, eps):
    """Cases in which ONE state is outside the goal box."""
    gx1, gy1, gx2, gy2 = env.goal
    return [[(0, -INF, gx1 - eps)], [(0, gx2 + eps, INF)], [(1, -INF, gy1 - eps)], [(1, gy2 + eps, INF)]]


# --------------------------------------------------------------------------- CEGIS loss terms (numpy or torch)
def ce_penalties(env, states, controls, xp):
    """
    states (B, T+1, d), controls (B, T, c). Returns {active constraint: tensor (B,)} - a real number per trajectory that is
    0 when the constraint holds with the safety buffer and grows with the amount of violation (distance units divided by the
    workspace size, control units divided by the control range). Minimising it moves the trajectory to the safe side.
    """
    active = active_constraints(env)
    xmin, ymin, xmax, ymax = env.workspace
    scale = max(xmax - xmin, ymax - ymin)
    sbuf, cbuf = config.CE_STATE_BUFFER, config.CE_CONTROL_BUFFER
    x, y = states[..., 0], states[..., 1]
    out = {}
    if "control" in active:
        total = 0.0
        for j in range(len(env.ctrl_lo)):
            span = float(env.ctrl_hi[j] - env.ctrl_lo[j])
            below = _relu(xp, env.ctrl_lo[j] - controls[..., j] + cbuf)
            above = _relu(xp, controls[..., j] - env.ctrl_hi[j] + cbuf)
            total = total + _tsum(xp, below + above) / span
        out["control"] = total
    if "obstacle" in active:
        total = 0.0
        for depth in obstacle_depths(env, x, y, xp):
            total = total + _tsum(xp, _relu(xp, depth + sbuf)) / scale
        out["obstacle"] = total
    if "workspace" in active:
        out["workspace"] = _tsum(xp, _relu(xp, workspace_degree(env, x, y, xp) + sbuf)) / scale
    if "theta" in active:
        span = float(env.theta_bounds[1] - env.theta_bounds[0])
        out["theta"] = _tsum(xp, _relu(xp, theta_degree(env, states[..., 2], xp) + cbuf)) / span
    if "goal" in active:
        d = goal_degree(env, x, y, xp)
        if config.GOAL_MODE == "final":
            d = d[:, -1]
        else:   # "any": only the closest state to the goal matters
            d = _tmin(xp, d)
        out["goal"] = _relu(xp, d + sbuf) / scale
    return out
