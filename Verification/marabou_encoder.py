"""Builds the verification problem for Marabou: the closed loop (policy network + true dynamics) unrolled over k steps,
starting from a box of start states.

Layers of the file
  Query      a small Marabou-independent description of the constraint system (variables with bounds, affine / ReLU / band /
             wrap constraints, and the property = the violation to look for). It propagates interval bounds while it is built,
             can evaluate itself numerically (used by selftest.py) and is converted to a Marabou query only in to_marabou().
  blocks     add_network (exact Linear / ReLU), pwl / trig / product (cos, sin and products of two variables as piecewise-linear
             BANDS that always contain the true value), SimpleSpec / UnicycleSpec (one dynamics step each).
  build_rollout  chains the blocks into the closed loop of the whole horizon.

Soundness: the simple model is encoded exactly. For the unicycle the bands enclose cos / sin / v*cos / v*sin, so a Marabou
UNSAT is a real proof; a Marabou SAT is only trusted after it is replayed on the true dynamics (see verifier.check_job).
"""
import math
import numpy as np

import config
import constraints

# Constants. SLACK widens propagated bounds slightly outward so that floating-point round-off can never exclude a real solution.
PI = math.pi
SLACK = 1e-7     # outward widening of propagated bounds (keeps floating point round-off from excluding real solutions)


# =====================================================================================================
# Query: a small, Marabou-independent description of the constraint system.
# It is converted to a Marabou query only inside to_marabou(), so everything else can be tested without Marabou.
# =====================================================================================================
# Collects variables and constraints. Every constraint that creates a new variable also propagates interval bounds to it.
class Query:
    # n = number of variables; _lo / _hi = their bounds (grown on demand); ops = constraints in creation order;
    # property = the violation to search for; infeasible = the bounds already prove there is no solution.
    def __init__(self):
        self.n = 0
        self._lo = np.full(1024, -np.inf)
        self._hi = np.full(1024, np.inf)
        self.ops = []              # constraints in creation order: affine / relu / band / wrap
        self.property = []         # conjunction of disjunctions; disjunction = [case]; case = [(var, lower, upper), ...]
        self.infeasible = False    # the propagated bounds already prove that the query has no solution
        self.start_vars = None
        self.output_vars = None

    # ---- variables and bounds
    @property
    def lo(self):
        return self._lo[:self.n]

    @property
    def hi(self):
        return self._hi[:self.n]

    # Creates variables with the given bounds and returns their indices.
    def new_vars(self, lo, hi):
        lo, hi = np.atleast_1d(np.asarray(lo, dtype=float)), np.atleast_1d(np.asarray(hi, dtype=float))
        k = len(lo)
        while self.n + k > len(self._lo):
            self._lo = np.concatenate([self._lo, np.full(len(self._lo), -np.inf)])
            self._hi = np.concatenate([self._hi, np.full(len(self._hi), np.inf)])
        idx = np.arange(self.n, self.n + k)
        self._lo[idx], self._hi[idx] = lo, hi
        self.n += k
        return idx

    # Intersects the bounds of existing variables with [lo, hi]; empty bounds mark the query as infeasible.
    def tighten(self, idx, lo, hi):
        idx = np.atleast_1d(idx)
        self._lo[idx] = np.maximum(self._lo[idx], lo)
        self._hi[idx] = np.minimum(self._hi[idx], hi)
        if np.any(self._lo[idx] > self._hi[idx]):
            self.infeasible = True

    # ---- constraints (each one also propagates interval bounds to the variable it creates)
    # Interval arithmetic: positive weights take the matching bound, negative weights the opposite bound.
    def affine(self, in_idx, W, b):
        """New variables out = W @ in + b."""
        in_idx, W, b = np.atleast_1d(in_idx), np.atleast_2d(np.asarray(W, dtype=float)), np.atleast_1d(np.asarray(b, dtype=float))
        lo_in, hi_in = self._lo[in_idx], self._hi[in_idx]
        Wp, Wn = np.maximum(W, 0.0), np.minimum(W, 0.0)
        out = self.new_vars(Wp @ lo_in + Wn @ hi_in + b - SLACK, Wp @ hi_in + Wn @ lo_in + b + SLACK)
        self.ops.append(("affine", out, in_idx, W, b))
        return out

    # New variables out = max(in, 0) (a ReLU constraint for Marabou), with propagated bounds.
    def relu(self, in_idx):
        in_idx = np.atleast_1d(in_idx)
        out = self.new_vars(np.maximum(self._lo[in_idx], 0.0), np.maximum(self._hi[in_idx], 0.0))
        self.ops.append(("relu", in_idx, out))
        return out

    def band(self, mid, e_lo, e_hi, lo, hi, exact):
        """New variable out with mid - e_lo <= out <= mid + e_hi, also inside [lo, hi]. exact(values) is only used by evaluate()."""
        out = self.new_vars(max(lo, self._lo[mid] - e_lo), min(hi, self._hi[mid] + e_hi))
        self.ops.append(("band", int(out[0]), int(mid), e_lo, e_hi, exact))
        return int(out[0])

    # Earlier version of wrap() without the safeguards for unbounded / very wide inputs; kept for reference (replaced by the version below).
    # def wrap(self, a):
    #     """New variable out = wrap_angle(a) in [-pi, pi]. The wrap is a disjunction over the 2*pi shifts that can occur."""
    #     lo, hi = self._lo[a], self._hi[a]
    # ms = the multiples of 2*pi that can occur: out = a - 2*pi*m for the m with a in [-pi + 2*pi*m, pi + 2*pi*m].
    # One possible shift = a plain affine map; several = a disjunction that Marabou splits on.
    #     ms = list(range(math.ceil((lo - PI) / (2 * PI)), math.floor((hi + PI) / (2 * PI)) + 1))
    #     if len(ms) == 1:
    #         out = self.affine([a], [[1.0]], [-2 * PI * ms[0]])
    #         self.tighten(out, -PI - SLACK, PI + SLACK)
    #         return int(out[0])
    #     out = self.new_vars(-PI - SLACK, PI + SLACK)
    #     self.ops.append(("wrap", int(out[0]), int(a), ms))
    #     return int(out[0])

    def wrap(self, a):
        """New variable out = wrap_angle(a) in [-pi, pi]. The wrap is a disjunction over the 2*pi shifts that can occur."""
        # Infinite or very wide ranges are limited to +-4*pi first (the shifts outside this range cannot matter for the bounded start region).
        lo, hi = self._lo[a], self._hi[a]
        if not math.isfinite(lo) or not math.isfinite(hi) or (hi - lo > 6 * PI):
            lo = max(lo if math.isfinite(lo) else -4 * PI, -4 * PI)
            hi = min(hi if math.isfinite(hi) else 4 * PI, 4 * PI)
        ms = list(range(math.ceil((lo - PI) / (2 * PI)), math.floor((hi + PI) / (2 * PI)) + 1))
        if len(ms) > 10:
            mid = 0.5 * (lo + hi)
            ms = list(range(math.floor((mid - 2 * PI) / (2 * PI)), math.ceil((mid + 2 * PI) / (2 * PI)) + 1))
        if len(ms) == 1:
            out = self.affine([a], [[1.0]], [-2 * PI * ms[0]])
            self.tighten(out, -PI - SLACK, PI + SLACK)
            return int(out[0])
        out = self.new_vars(-PI - SLACK, PI + SLACK)
        self.ops.append(("wrap", int(out[0]), int(a), ms))
        return int(out[0])

    # ---- property (what Marabou has to find: a violation)
    def add_property(self, cases, prune):
        """cases: list of cases (a disjunction). With pruning, cases that the propagated bounds exclude are dropped."""
        if prune:
            cases = [c for c in cases if all(lo <= self._hi[v] + 1e-9 and hi >= self._lo[v] - 1e-9 for v, lo, hi in c)]
            if not cases:
                self.infeasible = True
        if cases:
            self.property.append(cases)

    # ---- numeric evaluation of the constraint system (used by selftest.py)
    # Numerically evaluates the whole constraint system for a given start state (the band constraints are replaced by the exact
    # function they enclose). Used by selftest.py to check that the encoding contains the true trajectory.
    def evaluate(self, start_values):
        val = np.full(self.n, np.nan)
        val[self.start_vars] = start_values
        for op in self.ops:
            if op[0] == "affine":
                _, out, inp, W, b = op
                val[out] = W @ val[inp] + b
            elif op[0] == "relu":
                val[op[2]] = np.maximum(val[op[1]], 0.0)
            elif op[0] == "band":
                val[op[1]] = op[5](val)
            else:
                val[op[1]] = np.arctan2(np.sin(val[op[2]]), np.cos(val[op[2]]))
        return val

    def violations_of(self, val, tol=1e-6):
        """Lists everything in the constraint system that the evaluated values do not satisfy (must be empty)."""
        bad = []
        if np.any(val < self.lo - tol) or np.any(val > self.hi + tol):
            bad.append("a variable is outside its propagated bounds")
        for op in self.ops:
            if op[0] == "band" and not (val[op[2]] - op[3] - tol <= val[op[1]] <= val[op[2]] + op[4] + tol):
                bad.append("a trig/product band does not contain the true value")
            if op[0] == "wrap" and not any(-PI + 2 * PI * m - tol <= val[op[2]] <= PI + 2 * PI * m + tol and abs(val[op[1]] - (val[op[2]] - 2 * PI * m)) < tol for m in op[3]):
                bad.append("the wrap disjunction does not contain the true value")
        return bad

    # ---- Marabou
    # Translates the Query into a MarabouNetwork: equations for affine / band constraints, ReLU and disjunction constraints,
    # variable bounds, and the property (a single case becomes plain bounds, several cases become a disjunction).
    def to_marabou(self):
        from maraboupy.MarabouNetwork import MarabouNetwork
        from maraboupy import MarabouCore, MarabouUtils
        EQ, LE, GE = MarabouCore.Equation.EQ, MarabouCore.Equation.LE, MarabouCore.Equation.GE

        def equation(kind, terms, scalar):
            e = MarabouUtils.Equation(kind)
            for coef, var in terms:
                e.addAddend(float(coef), int(var))
            e.setScalar(float(scalar))
            return e

        # Marabou query is assembled from the stored operations in creation order.
        net = MarabouNetwork()
        net.numVars = self.n
        for op in self.ops:
            if op[0] == "affine":
                _, out, inp, W, b = op
                for i in range(len(out)):
                    nz = np.nonzero(W[i])[0]
                    net.addEquation(equation(EQ, [(-1.0, out[i])] + [(W[i, j], inp[j]) for j in nz], -b[i]))
            elif op[0] == "relu":
                for i, o in zip(op[1], op[2]):
                    net.addRelu(int(i), int(o))
            elif op[0] == "band":
                _, out, mid, e_lo, e_hi, _ = op
                net.addEquation(equation(LE, [(1.0, out), (-1.0, mid)], e_hi))
                net.addEquation(equation(GE, [(1.0, out), (-1.0, mid)], -e_lo))
            else:
                _, out, a, ms = op
                net.addDisjunctionConstraint([[equation(GE, [(1.0, a)], -PI + 2 * PI * m), equation(LE, [(1.0, a)], PI + 2 * PI * m),
                                               equation(EQ, [(1.0, out), (-1.0, a)], -2 * PI * m)] for m in ms])
        # The property: a single case is just a set of tighter variable bounds; several cases form a disjunction.
        lo, hi = self.lo.copy(), self.hi.copy()
        for cases in self.property:
            if len(cases) == 1:          # a single case is just a set of bounds
                for v, c_lo, c_hi in cases[0]:
                    lo[v], hi[v] = max(lo[v], c_lo), min(hi[v], c_hi)
            else:
                def bounds_as_equations(v, c_lo, c_hi):
                    eqs = []
                    if math.isfinite(c_lo):
                        eqs.append(equation(GE, [(1.0, v)], c_lo))
                    if math.isfinite(c_hi):
                        eqs.append(equation(LE, [(1.0, v)], c_hi))
                    return eqs
                net.addDisjunctionConstraint([[e for v, c_lo, c_hi in case for e in bounds_as_equations(v, c_lo, c_hi)] for case in cases])
        for v in range(self.n):
            if math.isfinite(lo[v]):
                net.setLowerBound(v, float(lo[v]))
            if math.isfinite(hi[v]):
                net.setUpperBound(v, float(hi[v]))
        # Marabou needs input / output variable lists to be set (their values are read back through the start variables).
        net.inputVars = [np.array([self.start_vars])]
        net.outputVars = [np.array([self.output_vars])]
        return net

    def solve(self, timeout):
        """Runs Marabou. Returns (status, values): status is 'sat', 'unsat', 'timeout' or 'unknown'."""
        from maraboupy import Marabou
        net = self.to_marabou()
        # Run Marabou; with more than one worker, split-and-conquer (snc) is used. The timeout is in whole seconds (at least 1).
        options = Marabou.createOptions(numWorkers=config.NUM_WORKERS, timeoutInSeconds=max(1, int(timeout)), verbosity=0,
                                        snc=config.NUM_WORKERS > 1)
        code, values, _ = net.solve(verbose=False, options=options)
        code = str(code).lower()
        return {"sat": "sat", "unsat": "unsat", "timeout": "timeout"}.get(code, "unknown"), values


# =====================================================================================================
# Building blocks: the policy network, the piecewise-linear bands, and the two dynamics
# =====================================================================================================
class Encoding:
    """Spacing of the piecewise-linear bands. refined(level) halves the spacing level times (band width / 4 each time)."""

    def __init__(self, level=0):
        self.level = level
        self.trig_step = config.TRIG_STEP / 2 ** level
        self.square_step = config.SQUARE_STEP / 2 ** level
        self.max_knots = config.MAX_PWL_KNOTS * 2 ** level


def add_network(q, layers, in_vars):
    """One copy of the policy: exact Linear / ReLU constraints. Returns the control variables."""
    a = np.asarray(in_vars)
    for W, b, relu in layers:
        a = q.affine(a, W, b)
        if relu:
            a = q.relu(a)
    return a


def range_trig(fn, lo, hi):
    """Exact range of cos or sin over [lo, hi]."""
    # A range of at least one full period covers the whole interval [-1, 1].
    if hi - lo >= 2 * PI:
        return -1.0, 1.0
    # The extreme values of cos / sin inside [lo, hi] are at the ends or at the points where the derivative is 0.
    shift = 0.0 if fn is np.cos else PI / 2          # cos has its extrema at k*pi, sin at pi/2 + k*pi
    pts = [lo, hi] + [shift + k * PI for k in range(math.ceil((lo - shift) / PI), math.floor((hi - shift) / PI) + 1)]
    vals = fn(np.array(pts))
    return float(vals.min()), float(vals.max())


def pwl(q, var, fn, step, enc):
    """
    Piecewise-linear interpolant p of fn on the bounds of `var`, written with ReLUs:
        p(t) = f0 + s0 (t - t0) + sum_i (s_i - s_{i-1}) relu(t - t_i)
    Returns (variable holding p(var), knot spacing h). For a function with |f''| <= K: |fn - p| <= K h^2 / 8.
    """
    lo, hi = float(q.lo[var]) - 1e-6, float(q.hi[var]) + 1e-6
    # Number of linear pieces: about (range / step), at least 1 and at most max_knots (a wider band is the price, still sound).
    n = int(min(enc.max_knots, max(1, math.ceil((hi - lo) / step))))
    t = np.linspace(lo, hi, n + 1)
    f = fn(t)
    # Slopes of the pieces; the interpolant is written as f0 + s0*(t - t0) + sum (s_i - s_(i-1)) * relu(t - t_i).
    s = np.diff(f) / np.diff(t)
    if n > 1:
        r = q.relu(q.affine([var], np.ones((n - 1, 1)), -t[1:n]))
        mid = q.affine(np.concatenate([[var], r]), np.concatenate([[s[0]], np.diff(s)])[None, :], [f[0] - s[0] * t[0]])
    else:
        mid = q.affine([var], [[s[0]]], [f[0] - s[0] * t[0]])
    return int(mid[0]), (hi - lo) / n


def trig(q, theta, fn, enc):
    """New variable c with |c - fn(theta)| <= band, for fn = np.cos or np.sin (|fn''| <= 1)."""
    mid, h = pwl(q, theta, fn, enc.trig_step, enc)
    e = h * h / 8 + 1e-9
    r_lo, r_hi = range_trig(fn, float(q.lo[theta]), float(q.hi[theta]))
    return q.band(mid, e, e, max(-1.0, r_lo - SLACK), min(1.0, r_hi + SLACK), exact=lambda val: fn(val[theta]))


def product(q, v, c, enc):
    """
    New variable w with w ~ v * c, from v*c = ((v + c)^2 - (v - c)^2) / 4.
    A square z^2 lies in [p(z) - h^2/4, p(z)] (p = chord interpolant), so w lies in a band of half-width about h^2/16.
    """
    # v*c = ((v + c)^2 - (v - c)^2) / 4: only squares are needed, and a square is a one-variable function that can be interpolated.
    z1 = int(q.affine([v, c], [[1.0, 1.0]], [0.0])[0])
    z2 = int(q.affine([v, c], [[1.0, -1.0]], [0.0])[0])
    m1, h1 = pwl(q, z1, np.square, enc.square_step, enc)
    m2, h2 = pwl(q, z2, np.square, enc.square_step, enc)
    e1, e2 = h1 * h1 / 4 + 1e-9, h2 * h2 / 4 + 1e-9
    mid = int(q.affine([m1, m2], [[0.25, -0.25]], [0.0])[0])
    # Range of the product from the four corners of the box [v] x [c].
    corners = [q.lo[v] * q.lo[c], q.lo[v] * q.hi[c], q.hi[v] * q.lo[c], q.hi[v] * q.hi[c]]
    return q.band(mid, e1 / 4, e2 / 4, min(corners) - SLACK, max(corners) + SLACK, exact=lambda val: val[v] * val[c])


class SimpleSpec:
    """x' = x + u * dt: exact, nothing is approximated."""

    def step(self, q, env, s, u, enc):
        # Next state = [x + vx*dt, y + vy*dt] as one linear map of the vector (x, y, vx, vy).
        W = [[1.0, 0.0, env.dt, 0.0], [0.0, 1.0, 0.0, env.dt]]
        return q.affine(np.concatenate([s, u]), W, [0.0, 0.0])


class UnicycleSpec:
    """x' = x + v cos(theta) dt, y' = y + v sin(theta) dt, theta' = wrap(theta + omega dt), with cos, sin and products as bands."""

    def step(self, q, env, s, u, enc):
        x, y, theta = (int(i) for i in s)
        v, omega = int(u[0]), int(u[1])
        # wx ~ v*cos(theta), wy ~ v*sin(theta) as bands (see trig / product).
        wx = product(q, v, trig(q, theta, np.cos, enc), enc)
        wy = product(q, v, trig(q, theta, np.sin, enc), enc)
        x_next = int(q.affine([x, wx], [[1.0, env.dt]], [0.0])[0])
        y_next = int(q.affine([y, wy], [[1.0, env.dt]], [0.0])[0])
        a = int(q.affine([theta, omega], [[1.0, env.dt]], [0.0])[0])
        # The new heading theta + omega*dt is wrapped to [-pi, pi] like in the true dynamics.
        return np.array([x_next, y_next, q.wrap(a)])


# Returns the dynamics description that belongs to the model name.
def get_spec(model):
    return SimpleSpec() if model == "simple" else UnicycleSpec()


def build_rollout(env, layers, spec, cell, n_steps, n_controls, enc):
    """
    Closed loop from a start cell: states s_0..s_n_steps and controls u_0..u_(n_controls-1), where u_i = policy(s_i) and
    s_(i+1) = f(s_i, u_i) with the true dynamics (the same function the prediction scripts use).
    Controls that were applied before the property step are assumed to be inside their bounds (up to PROPERTY_EPS): if a
    trajectory breaks them earlier, the query of that earlier step reports it, so no unsafe trajectory is lost.
    """
    q = Query()
    # Start-state variables, bounded by the start cell.
    s = q.new_vars(cell[0], cell[1])
    q.start_vars = s
    states, controls = [s], []
    assume = "control" in constraints.active_constraints(env)
    eps = config.PROPERTY_EPS
    for i in range(n_controls):
        # Closed loop: control u_i = policy(s_i); then (for i < n_steps) s_(i+1) = f(s_i, u_i).
        u = add_network(q, layers, states[i])
        controls.append(u)
        if i < n_steps:
            if assume:
                # Controls applied before the property step are assumed to be within their bounds (an earlier violation is caught by the
                # query of the earlier step); this keeps the bounds tight.
                q.tighten(u, np.array(env.ctrl_lo) - eps, np.array(env.ctrl_hi) + eps)
            states.append(spec.step(q, env, states[i], u, enc))
    q.output_vars = controls[-1] if controls else s
    return q, states, controls
