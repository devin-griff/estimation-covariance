# =============================================================================
# Estimation Covariance: a Streamlit tutorial app.
#
# Fit a two-parameter decay model to noisy data, then ask the solver how
# certain the fitted parameters are. The estimation problem is
#
#   minimize_{A,k}  sum_i r_i^2
#   subject to      r_i = y_i - A exp(-k x_i)
#
# and the uncertainty comes out of the SAME solve. `declare_fitted` flags
# which variables are the estimated parameters, `declare_residual` flags the
# container holding the residuals, and `covariance(m)` then reads the
# parameter covariance off the KKT factorization the solve already produced
# (one backsolve per parameter, no second optimization).
#
# The app's thesis is a cost comparison the user can watch: the covariance
# costs about ten milliseconds, while confirming the same ellipse by brute
# force costs one solve per Monte Carlo draw. Both timings are on screen.
#
# Library roadmap:
#   - streamlit    : UI framework. Each interaction reruns this script
#                     top-to-bottom; persistent values live in session_state.
#   - pyomo        : algebraic modeling layer for the estimation NLP.
#   - pounce       : the NLP solver AND the covariance primitive. Ships as a
#                     pip wheel (`pyomo-pounce`), bundles its own binary, and
#                     needs no license.
#   - numpy        : data generation, eigendecomposition, ellipse geometry.
#   - pandas/altair: the three panels.
#
# Thread affinity (the one non-obvious constraint):
#   pounce holds its factorization in a Rust object that may only be used or
#   DROPPED by the thread that created it. Streamlit runs every rerun on a
#   fresh script thread, so a model left to the cyclic garbage collector can
#   be dropped on a later rerun's thread, raising "PySolver is unsendable".
#   Every solve here therefore drops its own model before returning (see
#   `_solve_and_extract`). That is also why this app needs no worker thread:
#   unlike the CSTR sensitivity demo, it never reuses a factorization across
#   reruns, so there is nothing to keep alive between clicks.
#
# File roadmap (matching the section banners below):
#   1. Constants   : palette, defaults, statistical critical values.
#   2. Solver      : model build, the fit + covariance call, Monte Carlo.
#   3. Geometry    : covariance matrix -> ellipse, box, marginal intervals.
#   4. State       : session_state init, seed re-rolls, staleness.
#   5. Charts      : the three panels of the main row.
#   6. LaTeX       : formulation tab content.
#   7. CSS         : template-family style tweaks.
#   8. Tabs        : render_main / render_formulation / render_logs.
#   9. Main        : page config, sidebar, tab assembly.
# =============================================================================

import base64
import contextlib
import gc
import io
import threading
import time
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import streamlit as st

# Registers `pounce` with pyo.SolverFactory via decorator side effect; the
# wheel also bundles the solver binary, so no system install is required.
import pyomo_pounce  # noqa: F401
from pyomo_pounce import covariance, declare_fitted, declare_residual

# ── 1. Constants ─────────────────────────────────────────────────────────────

# Chi-square critical value with 2 degrees of freedom at 95%. This is the
# level set that makes the ellipse a JOINT 95% region for (A, k).
CHI2_95_2DOF = 5.991
# Normal critical value at 95%, for a SINGLE parameter's marginal interval.
# sqrt(5.991) = 2.4477 > 1.96, which is why the joint region's shadow on one
# axis is wider than that parameter's own confidence interval.
Z_95 = 1.959964

# Wong (2011) palette, the family standard. Each covariance entry is keyed to
# the geometric feature it controls in the parameter-space panel.
COLOR_VAR_A = "#0072B2"   # var(A): the box's horizontal half-width
COLOR_VAR_K = "#E69F00"   # var(k): the box's vertical half-width
COLOR_COV = "#009E73"     # cov(A,k): the tilt, and the true ellipse
COLOR_TRUTH = "#D55E00"   # the parameter values the user chose
COLOR_MC = "#7f7f7f"      # Monte Carlo refits
COLOR_UNCORR = "#9ca3af"  # the same matrix with its off-diagonal zeroed

DEFAULTS = {
    "a_true": 2.0,
    "k_true": 1.3,
    "sigma": 0.05,
    "x_range": (0.0, 3.0),
    "n_pts": 20,
    "seed": 7,
    "n_draws": 200,
    "mc_seed": 0,
}

# Monte Carlo draw bounds. 200 draws reproduce the calibration result to
# within binomial noise (measured 95.5% coverage at 200, 96.4% at 500), and
# the ceiling keeps one run from monopolizing a shared machine: every draw is
# a full NLP solve, roughly 80 ms locally and slower on a small cloud vCPU.
MC_MIN, MC_MAX, MC_STEP = 50, 500, 50


# ── 2. Solver ────────────────────────────────────────────────────────────────

# One solve at a time per machine: every Streamlit session runs app.py in
# its own thread of this one process, and the solver-log capture redirects
# process-global stdout. Overlapping captures corrupt each other and fail
# both solves, so every solve serializes behind one process-wide lock.
# The lock must come from st.cache_resource: Streamlit re-executes this
# script per rerun in a fresh namespace, so a bare module-level Lock would
# be a new object every rerun and would serialize nothing. The Monte Carlo
# takes the lock per draw, not per run, so one long sweep does not block
# another visitor's single fit for its whole duration.
@st.cache_resource(show_spinner=False)
def _solve_lock():
    return threading.Lock()


def make_data(a_true, k_true, sigma, x_min, x_max, n_pts, seed):
    """The synthetic experiment: an exact decay plus Gaussian noise."""
    x = np.linspace(x_min, x_max, n_pts)
    clean = a_true * np.exp(-k_true * x)
    noise = np.random.default_rng(seed).standard_normal(n_pts)
    return x, clean + sigma * noise


def initial_guess(x, y):
    """A data-only warm start, from the log-linear form log y = log A - k x.

    The sum-of-squares surface for a decay is badly behaved far from the
    solution: from a fixed (A, k) = (1, 1) start on a short late window the
    solver runs off in an unbounded direction rather than converging. The
    notebook sidesteps this by starting at the true values, which an app
    cannot do honestly, since the truth is exactly what the fit is supposed
    to recover. A one-line linear regression on the log of the positive
    samples is the standard practitioner's answer and uses only the data.
    Noise can push late samples below zero, so those are simply skipped.
    """
    y = np.asarray(y, dtype=float)
    pos = y > 0
    if pos.sum() >= 2 and float(np.ptp(x[pos])) > 1e-9:
        slope, intercept = np.polyfit(x[pos], np.log(y[pos]), 1)
        a0, k0 = float(np.exp(intercept)), float(-slope)
        if np.isfinite(a0) and np.isfinite(k0) and 1e-6 < a0 < 1e4:
            return a0, float(np.clip(k0, -5.0, 50.0))
    return float(max(np.abs(y).max(), 1e-3)), 1.0


def build_model(x, y, declare):
    """The estimation NLP with EXPLICIT residual variables.

    The residuals are variables tied to the data by equality constraints
    rather than folded into the objective, because that is what gives
    `covariance()` a residual container to count and sum: the noise variance
    and the degrees of freedom are both derived from it, so nothing about the
    statistics is passed in by hand.

    `declare=False` is used for Monte Carlo refits, which only need the point
    estimate: skipping the declarations skips the covariance bookkeeping.
    """
    a0, k0 = initial_guess(x, y)
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(x) - 1)
    m.A = pyo.Var(initialize=a0)
    m.k = pyo.Var(initialize=k0)
    m.r = pyo.Var(m.I, initialize=0.0)

    @m.Constraint(m.I)
    def res(m, i):
        return m.r[i] == float(y[i]) - m.A * pyo.exp(-m.k * float(x[i]))

    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I), sense=pyo.minimize)
    if declare:
        declare_fitted(m.A, m.k)
        declare_residual(m.r)
    return m


def _solve_and_extract(x, y):
    """Solve once, read the covariance, return PLAIN data.

    Everything pounce-owned is created and destroyed inside this call, on the
    calling thread. The explicit drop plus collection at the end is not
    tidiness: a Pyomo model is a web of reference cycles, so without it the
    solver object waits for the cyclic collector and can be freed on a later
    rerun's thread, which raises "PySolver is unsendable".
    """
    buf = io.StringIO()
    m = build_model(x, y, declare=True)
    cov = None
    try:
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                results = pyo.SolverFactory("pounce").solve(m, tee=True)
            term = str(results.solver.termination_condition)
        except Exception as exc:
            # A noise-dominated window can defeat the solve outright. Report
            # it as a status rather than letting a traceback reach the page.
            return {"status": "failed", "detail": str(exc),
                    "log": buf.getvalue()}
        if term != "optimal":
            return {"status": term, "log": buf.getvalue()}

        # pounce warns when the held KKT factor carried inertia-correction
        # perturbations, meaning the covariance is regularized rather than
        # exact. That is its own identifiability diagnostic and fires exactly
        # in the cases this app is built to show, so it is surfaced in the UI
        # rather than swallowed into the terminal.
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cov = covariance(m)
        cov_seconds = time.perf_counter() - t0
        notes = [str(w.message) for w in caught]

        C = np.array(cov.matrix, dtype=float)
        out = {
            "status": "optimal",
            "A": float(pyo.value(m.A)),
            "k": float(pyo.value(m.k)),
            "C": C.tolist(),
            "se": [float(cov.std_err[m.A]), float(cov.std_err[m.k])],
            "corr": float(cov.correlation[m.A, m.k]),
            "sigma_hat": float(np.sqrt(cov.sigma_sq)),
            "cov_seconds": cov_seconds,
            "notes": notes,
            "log": buf.getvalue(),
        }
        del results
        return out
    finally:
        del cov, m
        gc.collect()


@st.cache_data(show_spinner=False)
def fit(a_true, k_true, sigma, x_min, x_max, n_pts, seed):
    """Cached fit for the current settings. Cheap enough (about 90 ms) to
    re-run automatically whenever a control moves, which is why this app has
    no Solve button. The cache means dragging a slider back to a previous
    value costs nothing."""
    x, y = make_data(a_true, k_true, sigma, x_min, x_max, n_pts, seed)
    with _solve_lock():
        out = _solve_and_extract(x, y)
    out["x"] = x.tolist()
    out["y"] = y.tolist()
    return out


def run_monte_carlo(a_true, k_true, sigma, x_min, x_max, n_pts,
                    n_draws, mc_seed, progress=None):
    """Refit `n_draws` independent synthetic datasets drawn from the same
    truth and noise level. This is the brute-force answer the covariance
    predicts analytically, and its cost is the point: one full solve per
    draw, against a single backsolve for the whole ellipse.

    Not cached: it sits behind an explicit button, and the result is stored
    in session_state so it can be cleared the moment a data setting changes.
    """
    x = np.linspace(x_min, x_max, n_pts)
    clean = a_true * np.exp(-k_true * x)
    fits = np.empty((n_draws, 2))
    opt = pyo.SolverFactory("pounce")
    failures = 0
    t0 = time.perf_counter()
    try:
        for j in range(n_draws):
            # Seeds are a deterministic function of (mc_seed, j), so a run is
            # reproducible, and the re-roll button moves to a fresh block.
            rng = np.random.default_rng((int(mc_seed) + 1) * 1_000_003 + j)
            y = clean + sigma * rng.standard_normal(n_pts)
            m = build_model(x, y, declare=False)
            with _solve_lock():
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    res = opt.solve(m)
            if str(res.solver.termination_condition) != "optimal":
                failures += 1
                fits[j] = (np.nan, np.nan)
            else:
                fits[j] = (pyo.value(m.A), pyo.value(m.k))
            m = None
            res = None
            if progress is not None and (j % 5 == 0 or j == n_draws - 1):
                progress.progress(
                    (j + 1) / n_draws,
                    text=f"Refitting dataset {j + 1} of {n_draws}...",
                )
    finally:
        del opt
        gc.collect()
    return {
        "fits": fits[~np.isnan(fits).any(axis=1)].tolist(),
        "failures": failures,
        "seconds": time.perf_counter() - t0,
        "n_draws": int(n_draws),
    }


# ── 3. Geometry ──────────────────────────────────────────────────────────────
#
# Everything drawn in the parameter-space panel is a direct reading of the
# covariance matrix C, for the 95% region {d : d' C^-1 d <= 5.991}:
#
#   - semi-axes      : sqrt(5.991 * lambda_i) along the eigenvectors of C
#   - bounding box   : half-widths sqrt(5.991) * se, from the DIAGONAL only
#   - marginal band  : half-widths 1.96 * se, the single-parameter interval
#
# The box depends on the diagonal alone, so zeroing the off-diagonal gives a
# different ellipse inscribed in the SAME box: the covariance term does not
# change either parameter's own spread, it only tilts the joint region. That
# pairing is what the panel is built to show.

def ellipse_points(center, C, n=241):
    """Points on the 95% ellipse for covariance C, centered at `center`."""
    evals, evecs = np.linalg.eigh(np.asarray(C, dtype=float))
    evals = np.clip(evals, 0.0, None)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    unit = np.stack([np.cos(th), np.sin(th)])
    pts = (evecs @ (np.sqrt(CHI2_95_2DOF * evals)[:, None] * unit)).T
    return pts + np.asarray(center, dtype=float)


def eigen_summary(C):
    """Eigenvalues ascending plus the dominant (sloppiest) direction. When
    the ratio is large the fit pins one combination of the parameters far
    better than either parameter alone, which is what unidentifiability looks
    like in these numbers."""
    evals, evecs = np.linalg.eigh(np.asarray(C, dtype=float))
    return evals, evecs[:, -1]


def coverage_fraction(fits, center, C):
    """Fraction of refits inside the 95% region centered at `center`."""
    d = np.asarray(fits, dtype=float) - np.asarray(center, dtype=float)
    q = np.einsum("ij,jk,ik->i", d, np.linalg.inv(np.asarray(C, dtype=float)), d)
    return float((q <= CHI2_95_2DOF).mean())


# ── 4. State ─────────────────────────────────────────────────────────────────

def init_state():
    for key, val in DEFAULTS.items():
        st.session_state.setdefault(key, val)
    st.session_state.setdefault("mc", None)
    st.session_state.setdefault("mc_sig", None)


def reroll_data():
    """Advance the dataset seed: a new noise draw, so a new fit and a new
    ellipse. Runs as a button callback, which is the one place a
    widget-backed session_state key may be assigned."""
    st.session_state.seed = int(st.session_state.seed) + 1


def reroll_mc():
    """Advance the Monte Carlo seed block: the same fit, a fresh cloud. Makes
    it visible that coverage is itself a random quantity hovering near 95%
    rather than a fixed property of the method."""
    st.session_state.mc_seed = int(st.session_state.mc_seed) + 1


def data_signature():
    """Everything that changes the dataset, and therefore invalidates a
    stored Monte Carlo cloud. A scatter drawn under one truth sitting beneath
    an ellipse computed under another would be actively misleading, so the
    cloud is dropped rather than redrawn."""
    ss = st.session_state
    return (ss.a_true, ss.k_true, ss.sigma, tuple(ss.x_range),
            ss.n_pts, ss.seed)


# ── 5. Charts ────────────────────────────────────────────────────────────────

PANEL_HEIGHT = 330
_AXIS = {"labelFontSize": 12, "titleFontSize": 13}


def data_panel(d, a_true, k_true):
    """Left panel: the fit in data space. The noisy sample, the fitted
    curve, and the curve the data were generated from."""
    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    xx = np.linspace(float(x.min()), float(x.max()), 240)
    curves = pd.concat([
        pd.DataFrame({"x": xx, "y": d["A"] * np.exp(-d["k"] * xx),
                      "series": "fitted"}),
        pd.DataFrame({"x": xx, "y": a_true * np.exp(-k_true * xx),
                      "series": "truth"}),
    ])
    scale = alt.Scale(domain=["fitted", "truth"],
                      range=[COLOR_COV, COLOR_TRUTH])
    lines = alt.Chart(curves).mark_line(size=2).encode(
        x=alt.X("x:Q", title="x"),
        y=alt.Y("y:Q", title="y"),
        color=alt.Color("series:N", scale=scale,
                        legend=alt.Legend(title=None, orient="top-right")),
        strokeDash=alt.StrokeDash("series:N", legend=None),
    )
    pts = alt.Chart(pd.DataFrame({"x": x, "y": y})).mark_circle(
        size=52, color="#111827", opacity=0.75,
    ).encode(
        x="x:Q", y="y:Q",
        tooltip=[alt.Tooltip("x:Q", format=".3f"),
                 alt.Tooltip("y:Q", format=".3f")],
    )
    return alt.layer(lines, pts).properties(
        height=PANEL_HEIGHT).configure_axis(**_AXIS)


def _fmt(v):
    """Compact fixed-width rendering for covariance entries, which run small
    (1e-3 and below) but are not always tiny."""
    a = abs(v)
    if a >= 1e-2 or a == 0.0:
        return f"{v:.4f}"
    return f"{v:.2e}"


def matrix_panel(C, title, keyed=True):
    """Middle panel: the covariance matrix itself.

    Cells are colored by the ROLE each entry plays in the right-hand panel
    rather than by magnitude, so the two panels read as one object: the
    diagonal entries set the bounding box, the shared off-diagonal entry
    sets the tilt.
    """
    C = np.asarray(C, dtype=float)
    names = ["A", "k"]
    rows = []
    for i in range(2):
        for j in range(2):
            role = "var(A)" if i == j == 0 else "var(k)" if i == j == 1 else "cov"
            rows.append({
                "row": names[i], "col": names[j], "val": float(C[i, j]),
                "label": _fmt(C[i, j]),
                "role": role if keyed else "plain",
            })
    df = pd.DataFrame(rows)
    scale = alt.Scale(
        domain=["var(A)", "var(k)", "cov", "plain"],
        range=[COLOR_VAR_A, COLOR_VAR_K, COLOR_COV, "#94a3b8"],
    )
    base = alt.Chart(df).encode(
        x=alt.X("col:N", title=None, sort=names,
                axis=alt.Axis(labelFontSize=13, orient="top")),
        y=alt.Y("row:N", title=None, sort=names,
                axis=alt.Axis(labelFontSize=13)),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=3, opacity=0.9).encode(
        color=alt.Color("role:N", scale=scale, legend=None),
        tooltip=[alt.Tooltip("val:Q", format=".6g", title="value")],
    )
    text = base.mark_text(fontSize=12, fontWeight="bold", color="white").encode(
        text="label:N")
    return alt.layer(cells, text).properties(
        height=190, title=alt.TitleParams(title, fontSize=12, anchor="start"),
    )


def parameter_panel(d, a_true, k_true, mc):
    """Right panel: the same matrix, drawn.

    Layers, in the order they teach:
      - the dashed box, set by the diagonal entries alone;
      - the axis-aligned ellipse the matrix would give with no covariance;
      - the actual, tilted ellipse;
      - the marginal 95% crosshair, deliberately shorter than the box;
      - the Monte Carlo cloud and a truth-centered reference ellipse, once
        a sweep has been run.
    """
    C = np.asarray(d["C"], dtype=float)
    se = np.asarray(d["se"], dtype=float)
    center = np.array([d["A"], d["k"]], dtype=float)
    half = np.sqrt(CHI2_95_2DOF) * se

    # Every layer must declare the same non-zero-anchored scales. Altair
    # merges scales across a layered chart, so letting one layer default to
    # zero=True would drag both axes back to the origin and flatten the
    # ellipse into an unreadable speck.
    def _x(title="A"):
        return alt.X("A:Q", title=title, scale=alt.Scale(zero=False))

    def _y(title="k"):
        return alt.Y("k:Q", title=title, scale=alt.Scale(zero=False))

    ell = ellipse_points(center, C)
    ell_diag = ellipse_points(center, np.diag(np.diag(C)))
    curves = pd.concat([
        pd.DataFrame({"A": ell[:, 0], "k": ell[:, 1],
                      "series": "95% region"}),
        pd.DataFrame({"A": ell_diag[:, 0], "k": ell_diag[:, 1],
                      "series": "if uncorrelated"}),
    ])
    scale = alt.Scale(domain=["95% region", "if uncorrelated"],
                      range=[COLOR_COV, COLOR_UNCORR])

    layers = []

    box = pd.DataFrame({"a1": [center[0] - half[0]], "a2": [center[0] + half[0]],
                        "k1": [center[1] - half[1]], "k2": [center[1] + half[1]]})
    layers.append(alt.Chart(box).mark_rect(
        fillOpacity=0.0, stroke="#475569", strokeWidth=1.2, strokeDash=[5, 4],
    ).encode(
        x=alt.X("a1:Q", title="A", scale=alt.Scale(zero=False)), x2="a2:Q",
        y=alt.Y("k1:Q", title="k", scale=alt.Scale(zero=False)), y2="k2:Q",
    ))

    layers.append(alt.Chart(curves).mark_line(size=2).encode(
        x=_x(), y=_y(),
        color=alt.Color("series:N", scale=scale,
                        legend=alt.Legend(title=None, orient="top-right")),
        strokeDash=alt.StrokeDash("series:N", legend=None),
    ))

    # Marginal 95% intervals: 1.96 se, against the box's 2.45 se. The visible
    # gap is the point, so these are drawn through the center as a crosshair.
    marg = pd.DataFrame({
        "a1": [center[0] - Z_95 * se[0], center[0]],
        "a2": [center[0] + Z_95 * se[0], center[0]],
        "k1": [center[1], center[1] - Z_95 * se[1]],
        "k2": [center[1], center[1] + Z_95 * se[1]],
    })
    layers.append(alt.Chart(marg).mark_rule(
        color="#111827", strokeWidth=2, opacity=0.85,
    ).encode(
        x=alt.X("a1:Q", title="A", scale=alt.Scale(zero=False)), x2="a2:Q",
        y=alt.Y("k1:Q", title="k", scale=alt.Scale(zero=False)), y2="k2:Q",
    ))

    if mc and mc.get("fits"):
        pts = np.asarray(mc["fits"], dtype=float)
        layers.insert(0, alt.Chart(
            pd.DataFrame({"A": pts[:, 0], "k": pts[:, 1]})
        ).mark_circle(size=18, color=COLOR_MC, opacity=0.35).encode(
            x=_x(), y=_y(),
            tooltip=[alt.Tooltip("A:Q", format=".4f"),
                     alt.Tooltip("k:Q", format=".4f")],
        ))
        # The refits are sampled around the TRUTH, so the region they are
        # tested against is centered there, not on this one estimate.
        ref = ellipse_points(np.array([a_true, k_true]), C)
        layers.append(alt.Chart(
            pd.DataFrame({"A": ref[:, 0], "k": ref[:, 1]})
        ).mark_line(size=1.2, color=COLOR_TRUTH, strokeDash=[3, 3]).encode(
            x=_x(), y=_y()))

    layers.append(alt.Chart(
        pd.DataFrame({"A": [a_true], "k": [k_true]})
    ).mark_point(shape="cross", size=150, color=COLOR_TRUTH,
                 strokeWidth=3, filled=False).encode(x=_x(), y=_y()))
    layers.append(alt.Chart(
        pd.DataFrame({"A": [center[0]], "k": [center[1]]})
    ).mark_point(size=90, color="#111827", filled=True).encode(
        x=_x(), y=_y()))

    return alt.layer(*layers).properties(
        height=PANEL_HEIGHT).configure_axis(**_AXIS)


# ── 6. LaTeX ─────────────────────────────────────────────────────────────────

def render_formulation():
    st.markdown("#### The estimation problem")
    st.latex(r"""
    \begin{aligned}
    \min_{A,\;k}\quad & \sum_{i=1}^{n} r_i^{2} \\
    \text{s.t.}\quad  & r_i = y_i - A\,e^{-k x_i}, \qquad i = 1,\dots,n
    \end{aligned}
    """)
    st.markdown(
        "The residuals are explicit variables rather than an expression "
        "folded into the objective. That is what gives `covariance()` a "
        "container to read: the residual count and the sum of squares both "
        "come from it, so the degrees of freedom and the noise variance are "
        "derived rather than supplied."
    )

    st.markdown("#### Where the covariance comes from")
    st.markdown(
        "Two declarations mark the pieces, and the covariance is then read "
        "off the factorization the solve already produced. No second "
        "optimization, and no finite differencing: one backsolve per fitted "
        "parameter."
    )
    st.code(
        "declare_fitted(m.A, m.k)      # these are the estimated parameters\n"
        "declare_residual(m.r)         # this container holds the residuals\n"
        "\n"
        "pyo.SolverFactory('pounce').solve(m)\n"
        "cov = covariance(m)           # zero further arguments\n"
        "cov.matrix, cov.std_err, cov.correlation, cov.sigma_sq",
        language="python",
    )
    st.latex(r"""
    \hat\sigma^{2} = \frac{\sum_i r_i^{2}}{n - p},
    \qquad
    C = \hat\sigma^{2}\,\bigl(\text{reduced KKT}\bigr)^{-1},
    \qquad
    \mathrm{se}(\theta_j) = \sqrt{C_{jj}}
    """)

    st.markdown("#### From the matrix to the ellipse")
    st.markdown(
        "The 95% joint confidence region for the pair is the level set"
    )
    st.latex(r"""
    \Bigl\{\,\theta \;:\;
    (\theta-\hat\theta)^{\mathsf T} C^{-1} (\theta-\hat\theta)
    \;\le\; \chi^{2}_{2,\,0.95} = 5.991 \,\Bigr\}
    """)
    st.markdown(
        "Every mark in the right-hand panel is a reading of that one "
        "expression:"
    )
    st.latex(r"""
    \begin{aligned}
    \text{semi-axes:}   \quad & \sqrt{5.991\,\lambda_i}
                                \ \text{along the eigenvectors of } C \\
    \text{bounding box:}\quad & \pm\sqrt{5.991}\;\mathrm{se}(\theta_j)
                                = \pm 2.4477\,\mathrm{se}(\theta_j) \\
    \text{marginal interval:}\quad & \pm 1.96\,\mathrm{se}(\theta_j)
    \end{aligned}
    """)
    st.markdown(
        "The box depends only on the diagonal of $C$. Zeroing the "
        "off-diagonal therefore produces a different ellipse inscribed in "
        "**the same box**, which is exactly what the faint companion curve "
        "shows: the covariance term does not change either parameter's own "
        "spread, it only tilts and squeezes the joint region inside that "
        "box. It also explains the crosshair, which is shorter than the box "
        "by a factor of $2.4477/1.96 = 1.249$: reading a joint region off "
        "two individual error bars understates it."
    )

    st.markdown("#### What the Monte Carlo checks")
    st.markdown(
        "The ellipse claims to describe the sampling distribution of the "
        "estimates. The sweep tests that claim directly by refitting many "
        "independent datasets drawn from the same truth and noise, and "
        "counting how many land inside. The reported coverage is measured "
        "against the region centered at the **true** parameters, drawn as "
        "the dashed reference curve, because that is the distribution the "
        "refits are actually sampled from. Centering on one estimate instead "
        "would cover only about 78%, since two independent estimates differ "
        "with twice the covariance."
    )
    st.markdown(
        "Narrowing the x range to a short late window is worth trying: there "
        "the data pin the local value and slope, while $A$ is a long "
        "extrapolation back to $x=0$. The correlation runs to nearly 1 and "
        "the ellipse collapses to a sliver, which is what an unidentifiable "
        "pair looks like before any prediction is made."
    )


# ── 7. CSS ───────────────────────────────────────────────────────────────────

CSS = """
<style>
/* Streamlit's default block-container padding-top pushes the title below the
   fold on a 13" laptop; 2.5rem clears the sticky header without clipping. */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
}

/* Sidebar app: the logo sits in the sidebar's flow, and hiding Streamlit's
   sticky sidebar header keeps the collapse chrome from pushing it down. */
.home-logo-corner {
    display: block;
    margin: 0 0 0.75rem;
}
.home-logo-corner img {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    display: block;
}
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}
</style>
"""


# ── 8. Tabs ──────────────────────────────────────────────────────────────────

def render_main(d, a_true, k_true, sigma):
    if d["status"] != "optimal":
        x = np.asarray(d["x"], dtype=float)
        signal = float(np.mean(np.abs(a_true * np.exp(-k_true * x))))
        snr = signal / sigma if sigma > 0 else np.inf
        st.error(
            f"The fit did not converge (solver returned: {d['status']}).\n\n"
            f"Over this x range the signal averages {signal:.3g} while the "
            f"noise is σ = {sigma:g}, a signal-to-noise ratio of "
            f"{snr:.2f}. Below about 1 the data carry almost no information "
            "about A and k, and there is no curve left to fit. Widen the x "
            "range toward x = 0, lower the noise, or add data points."
        )
        return

    mc = st.session_state.mc
    C = np.asarray(d["C"], dtype=float)

    left, mid, right = st.columns([5, 3, 5], gap="medium")
    with left:
        st.markdown("**Data and fit**")
        st.altair_chart(data_panel(d, a_true, k_true), width="stretch")
    with mid:
        st.markdown("**Covariance**")
        st.altair_chart(matrix_panel(C, "reported by covariance()"),
                        width="stretch")
        if mc and mc.get("fits") and len(mc["fits"]) > 2:
            emp = np.cov(np.asarray(mc["fits"], dtype=float).T)
            st.altair_chart(
                matrix_panel(emp, f"empirical, {len(mc['fits'])} refits",
                             keyed=False),
                width="stretch")
    with right:
        st.markdown("**Parameter space**")
        st.altair_chart(parameter_panel(d, a_true, k_true, mc),
                        width="stretch")

    se = d["se"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A", f"{d['A']:.4f}", f"± {se[0]:.4f}  (true {a_true:g})",
              delta_color="off")
    c2.metric("k", f"{d['k']:.4f}", f"± {se[1]:.4f}  (true {k_true:g})",
              delta_color="off")
    c3.metric("correlation(A, k)", f"{d['corr']:+.3f}")
    c4.metric("σ̂", f"{d['sigma_hat']:.4f}", f"true {sigma:g}",
              delta_color="off")

    for note in d.get("notes") or []:
        st.warning(note, icon="⚠️")

    evals, sloppy = eigen_summary(C)
    ratio = evals[-1] / evals[0] if evals[0] > 0 else np.inf
    st.caption(
        f"Eigenvalues {evals[0]:.3e} and {evals[-1]:.3e} "
        f"(ratio {ratio:.1f}), dominant direction "
        f"({sloppy[0]:+.3f}, {sloppy[1]:+.3f}) in (A, k). "
        f"The box is ±{np.sqrt(CHI2_95_2DOF):.4f} standard errors wide; "
        f"the crosshair is ±{Z_95:.2f}."
    )

    st.divider()
    if mc and mc.get("fits"):
        fits = np.asarray(mc["fits"], dtype=float)
        cover = coverage_fraction(fits, [a_true, k_true], C)
        binom_se = 100.0 * np.sqrt(0.95 * 0.05 / max(len(fits), 1))
        m1, m2, m3 = st.columns(3)
        m1.metric("coverage of the 95% region", f"{100 * cover:.1f}%",
                  f"± {binom_se:.1f}% sampling noise", delta_color="off")
        m2.metric("covariance()", f"{1000 * d['cov_seconds']:.0f} ms",
                  "one backsolve per parameter", delta_color="off")
        m3.metric(f"Monte Carlo, {len(fits)} refits",
                  f"{mc['seconds']:.1f} s",
                  "one full solve per draw", delta_color="off")
        if mc["failures"]:
            st.warning(f"{mc['failures']} of {mc['n_draws']} refits did not "
                       "converge and were dropped.")
    else:
        st.info(
            "The ellipse above came from one solve. Run the Monte Carlo "
            "sweep in the sidebar to check it against independent refits.",
            icon="🎲",
        )


def render_logs(d):
    log = (d.get("log") or "").strip()
    if not log:
        st.info("No solver output captured for the last fit.")
        return
    st.caption(
        "pounce's output for the displayed fit. The Monte Carlo refits are "
        "solved with logging suppressed; only this fit's log is kept."
    )
    st.code(log, language="text")


# ── 9. Main ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Estimation Covariance",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)

# Home-link logo (sidebar variant). Embedded as a base64 data URL so loading
# the page makes no third-party network call. The 32px size lives on the img
# tag as well as in the CSS: the 512px source otherwise paints full size for
# a frame whenever it decodes before the stylesheet mounts.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.sidebar.markdown(
    '<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE: home" '
    'width="32" height="32" '
    'style="width:32px;height:32px;border-radius:4px;display:block" /></a>',
    unsafe_allow_html=True,
)

init_state()

st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; "
    "font-weight: 700;'>"
    "Estimation Covariance "
    "<a href='https://github.com/devin-griff/estimation-covariance' "
    "target='_blank' title='View source on GitHub' "
    "style='display: inline-block; vertical-align: 0.02em; "
    "margin: 0 0.35rem 0 0.1rem; color: inherit;'>"
    "<svg viewBox='0 0 16 16' width='20' height='20' fill='currentColor' "
    "aria-label='GitHub'>"
    "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17."
    "55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
    ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
    "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59."
    "82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
    ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51"
    ".56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1."
    "07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-"
    "8-8-8z'/></svg></a>"
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://github.com/jkitchin/pounce' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>pounce</a>"
    "</span></h2>",
    unsafe_allow_html=True,
)
st.caption("Parameter uncertainty from one solve, checked against Monte Carlo")

# ---- Sidebar: the experiment ----
st.sidebar.header("Truth")
st.sidebar.slider("A", 0.5, 5.0, key="a_true", step=0.1)
st.sidebar.slider("k", 0.2, 3.0, key="k_true", step=0.1)
st.sidebar.slider("noise σ", 0.005, 0.50, key="sigma", step=0.005,
                  format="%.3f")

st.sidebar.header("Sampling")
st.sidebar.slider(
    "x range", 0.0, 6.0, step=0.1, key="x_range",
    help="Narrow this to a short late window to see A and k become "
         "unidentifiable: the correlation runs to nearly 1 and the ellipse "
         "collapses to a sliver.",
)
st.sidebar.slider("data points", 5, 100, key="n_pts", step=1)
sc1, sc2 = st.sidebar.columns([2, 3])
sc1.number_input("seed", 0, 99999, key="seed", step=1)
sc2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
sc2.button("New dataset", on_click=reroll_data, width="stretch")

st.sidebar.header("Monte Carlo")
st.sidebar.slider("refits", MC_MIN, MC_MAX, key="n_draws", step=MC_STEP,
                  help="Each refit is a full NLP solve. 200 is enough to "
                       "confirm the coverage to within sampling noise.")
mc1, mc2 = st.sidebar.columns(2)
run_mc = mc1.button("Run sweep", type="primary", width="stretch")
mc2.button("New draws", on_click=reroll_mc, width="stretch")
mc_slot = st.sidebar.empty()

# A stored cloud belongs to the settings that produced it. If any of those
# changed, drop it rather than draw it under a freshly computed ellipse.
if st.session_state.mc_sig != data_signature():
    st.session_state.mc = None
    st.session_state.mc_sig = None

x_min, x_max = (float(v) for v in st.session_state.x_range)
if x_max - x_min < 0.05:
    st.error("The x range is too narrow to fit. Widen it in the sidebar.")
    st.stop()

d = fit(
    st.session_state.a_true, st.session_state.k_true, st.session_state.sigma,
    x_min, x_max, int(st.session_state.n_pts), int(st.session_state.seed),
)

if run_mc and d["status"] == "optimal":
    with mc_slot.container():
        bar = st.progress(0.0, text="Refitting...")
        result = run_monte_carlo(
            st.session_state.a_true, st.session_state.k_true,
            st.session_state.sigma, x_min, x_max,
            int(st.session_state.n_pts),
            int(st.session_state.n_draws), int(st.session_state.mc_seed),
            progress=bar,
        )
        bar.empty()
    st.session_state.mc = result
    st.session_state.mc_sig = data_signature()

tab_main, tab_formulation, tab_logs = st.tabs(
    ["📈 Estimate", "📐 Formulation", "📋 Logs"]
)
with tab_main:
    render_main(d, st.session_state.a_true, st.session_state.k_true,
                st.session_state.sigma)
with tab_formulation:
    render_formulation()
with tab_logs:
    render_logs(d)
