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
#   3. Geometry    : covariance matrix -> ellipse and principal axes.
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

# Wong (2011) palette, the family standard. Each covariance entry is keyed to
# the geometric feature it controls in the parameter-space panel.
COLOR_VAR_A = "#0072B2"   # var(A): A's own spread
COLOR_VAR_K = "#E69F00"   # var(k): k's own spread
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
    "seed": 15,
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
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cov = covariance(m)
        except Exception as exc:
            # In the no-signal regime the solve can converge while the KKT
            # backsolve fails: the information matrix is numerically
            # singular. Report it as a status, with the point estimate kept
            # so the data panel can still draw the fitted curve.
            return {"status": "singular", "detail": str(exc),
                    "A": float(pyo.value(m.A)), "k": float(pyo.value(m.k)),
                    "log": buf.getvalue()}
        cov_seconds = time.perf_counter() - t0
        # pounce's warning text is solver-speak with raw perturbation values.
        # The UI shows one plain sentence; the verbatim warnings go to the
        # Logs tab for anyone who wants the numbers.
        notes = []
        note_details = [str(w.message) for w in caught]
        if note_details:
            notes.append(
                "The covariance is approximate here: the two parameters are "
                "nearly indistinguishable over this window. Details on the "
                "Logs tab."
            )

        # sigma_hat, derived from the residuals with nothing passed by hand,
        # is reported as a metric: it is what the data alone say the noise
        # level is.
        sigma_hat = float(np.sqrt(cov.sigma_sq))

        # ONE matrix drives everything shown: the middle panel, the ellipse,
        # the axes, and the coverage count. It is exactly what covariance()
        # reports, scaled by sigma_hat from the residuals: the practitioner's
        # answer, which is the point of the exercise. The price is that
        # sigma_hat carries one dataset's chi-square fluctuation, so the
        # measured coverage sits a few points under 95% at small n and
        # swings dataset to dataset; the Formulation tab says so.
        C_plot = np.array(cov.matrix, dtype=float)
        se_plot = [float(cov.std_err[m.A]), float(cov.std_err[m.k])]

        out = {
            "status": "optimal",
            "A": float(pyo.value(m.A)),
            "k": float(pyo.value(m.k)),
            "C_plot": C_plot.tolist(),
            "se": se_plot,
            # Correlation is scale free, so rescaling the matrix by a
            # different noise variance leaves it unchanged.
            "corr": float(cov.correlation[m.A, m.k]),
            "sigma_hat": sigma_hat,
            "cov_seconds": cov_seconds,
            "notes": notes,
            "note_details": note_details,
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
    draw, against a single fit for the region on screen.

    Not cached: it sits behind an explicit button, and the result is stored
    in session_state so it can be cleared the moment a data setting changes.
    """
    x = np.linspace(x_min, x_max, n_pts)
    clean = a_true * np.exp(-k_true * x)
    fits = np.full((n_draws, 2), np.nan)
    opt = pyo.SolverFactory("pounce")
    failures = 0
    ssr_total = 0.0
    dof_total = 0
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
            else:
                a_j, k_j = pyo.value(m.A), pyo.value(m.k)
                fits[j] = (a_j, k_j)
                # The sweep's own noise estimate: pool every refit's residual
                # sum of squares, each with n - 2 degrees of freedom, exactly
                # the sigma-hat recipe applied to all the draws at once.
                r_j = y - a_j * np.exp(-k_j * x)
                ssr_total += float(r_j @ r_j)
                dof_total += n_pts - 2
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
    ok = ~np.isnan(fits).any(axis=1)
    return {
        "fits": fits[ok].tolist(),
        "sigma_mc": float(np.sqrt(ssr_total / dof_total)) if dof_total else None,
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
#
# Zeroing the off-diagonal gives the companion ellipse: the covariance term
# does not change either parameter's own spread, it only tilts the joint
# region. That pairing is what the panel is built to show.

def ellipse_points(center, C, n=241):
    """Points on the 95% ellipse for covariance C, centered at `center`."""
    evals, evecs = np.linalg.eigh(np.asarray(C, dtype=float))
    evals = np.clip(evals, 0.0, None)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    unit = np.stack([np.cos(th), np.sin(th)])
    pts = (evecs @ (np.sqrt(CHI2_95_2DOF * evals)[:, None] * unit)).T
    return pts + np.asarray(center, dtype=float)


# ── 4. State ─────────────────────────────────────────────────────────────────

def init_state():
    for key, val in DEFAULTS.items():
        st.session_state.setdefault(key, val)
    st.session_state.setdefault("mc", None)
    st.session_state.setdefault("mc_sig", None)


def data_signature():
    """Everything the stored Monte Carlo cloud depends on: a cloud generated
    under one truth sitting beneath a region computed under another would
    be actively misleading, so the cloud is dropped when any of these
    change. The MC seed and draw count are included because the stored
    sweep no longer matches the controls once either moves. The dataset
    seed is deliberately absent: it moves only YOUR
    dataset and its fitted region, not the cloud, so stepping it with a
    sweep on screen shows the sigma_hat-sized region wobbling against
    fixed refits."""
    ss = st.session_state
    return (ss.a_true, ss.k_true, ss.sigma, tuple(ss.x_range), ss.n_pts,
            ss.mc_seed, ss.n_draws)


# ── 5. Charts ────────────────────────────────────────────────────────────────

# Plotting-area size (a square side) shared by the data and parameter
# panels. 300 is the largest that fits both squares, the matrix, and the
# side legend across one row on a laptop-width window.
PANEL_HEIGHT = 300
_AXIS = {"labelFontSize": 12, "titleFontSize": 13}


def data_panel(d, a_true, k_true):
    """Left panel: the fit in data space. The noisy sample, the fitted
    curve, and the curve the data were generated from. When the fit failed
    there is no fitted curve, and the panel still renders: seeing the truth
    buried in noise IS the explanation of the failure."""
    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    xx = np.linspace(float(x.min()), float(x.max()), 240)
    frames = []
    domain, colors = [], []
    if d.get("A") is not None:
        frames.append(pd.DataFrame(
            {"x": xx, "y": d["A"] * np.exp(-d["k"] * xx), "series": "fitted"}))
        domain.append("fitted")
        colors.append(COLOR_COV)
    frames.append(pd.DataFrame(
        {"x": xx, "y": a_true * np.exp(-k_true * xx), "series": "truth"}))
    domain.append("truth")
    colors.append(COLOR_TRUTH)
    curves = pd.concat(frames)
    scale = alt.Scale(domain=domain, range=colors)
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
    # Same fixed plotting-area size as the parameter panel, so the two
    # bookend plots read as equals.
    return alt.layer(lines, pts).properties(
        width=PANEL_HEIGHT, height=PANEL_HEIGHT,
        autosize=alt.AutoSizeParams(type="pad", contains="padding"),
    ).configure_axis(**_AXIS)


def _fmt(v):
    """Compact fixed-width rendering for covariance entries, which run small
    (1e-3 and below) but are not always tiny."""
    a = abs(v)
    if a >= 1e-2 or a == 0.0:
        return f"{v:.4f}"
    return f"{v:.2e}"


def matrix_panel(C, title):
    """Middle panel: a covariance matrix as a 2x2 grid.

    Cells are colored by the ROLE each entry plays in the right-hand panel
    rather than by magnitude, so the panels read as one object: the
    diagonal entries set each parameter's spread, the shared off-diagonal
    entry sets the tilt. The Monte Carlo matrix uses the same key, so the
    two grids compare entry by entry at a glance.
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
                "role": role,
            })
    df = pd.DataFrame(rows)
    scale = alt.Scale(
        domain=["var(A)", "var(k)", "cov"],
        range=[COLOR_VAR_A, COLOR_VAR_K, COLOR_COV],
    )
    base = alt.Chart(df).encode(
        # labelAngle=0 keeps the column headers upright: Vega-Lite rotates
        # nominal x-axis labels a quarter turn by default.
        x=alt.X("col:N", title=None, sort=names,
                axis=alt.Axis(labelFontSize=14, orient="top", labelAngle=0)),
        y=alt.Y("row:N", title=None, sort=names,
                axis=alt.Axis(labelFontSize=14)),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=3, opacity=0.9).encode(
        color=alt.Color("role:N", scale=scale, legend=None),
        tooltip=[alt.Tooltip("val:Q", format=".6g", title="value")],
    )
    text = base.mark_text(fontSize=16, fontWeight="bold", color="white").encode(
        text="label:N")
    # Fixed and compact: two stacked matrices must fit beside the plot
    # panels without growing the row when the empirical one appears. The
    # rotated left-edge title names each matrix without a caption row.
    return alt.layer(cells, text).properties(
        width=140, height=110,
        title=alt.TitleParams(title, orient="left", anchor="middle",
                              fontSize=16, fontWeight="normal",
                              color="#31333f"),
        autosize=alt.AutoSizeParams(type="pad", contains="padding"),
    )


def parameter_panel(d, a_true, k_true, mc):
    """Right panel: the same matrix, drawn.

    Two curves from ONE matrix about ONE center:
      - the actual, tilted 95% region;
      - the axis-aligned ellipse the same matrix gives with its off-diagonal
        zeroed;
      - the principal axes of the region: the eigenvectors of C, each drawn
        with length sqrt(5.991 * eigenvalue);
      - the Monte Carlo cloud, once a sweep has been run.

    The region is exactly what a practitioner would draw: the reported
    matrix about THIS dataset's estimate. The truth is a fixed cross,
    inside the region for 95% of datasets. The Monte Carlo cloud scatters
    about the truth, not about the region's center, so the drawn region
    captures a replication's estimate about 78% of the time ON AVERAGE
    across datasets; any one dataset's capture depends on where its
    estimate happened to land.

    The panel is square with an EQUAL data span on both axes, so one data
    unit is the same number of pixels in A as in k. That is what makes the
    eigenvector geometry honest: the principal axes render truly orthogonal
    and the tilt angle is the real one, not an artifact of axis stretching.
    """
    C = np.asarray(d["C_plot"], dtype=float)
    se = np.sqrt(np.diag(C))
    center = np.array([d["A"], d["k"]], dtype=float)
    half = np.sqrt(CHI2_95_2DOF) * se

    # Shared square domain: every plotted item fits, both axes span the same
    # width. Scales merge across layers, so every layer declares these.
    extent = [half[0], half[1],
              abs(a_true - center[0]) * 1.3, abs(k_true - center[1]) * 1.3]
    if mc and mc.get("fits"):
        pts = np.asarray(mc["fits"], dtype=float)
        extent += [np.abs(pts[:, 0] - center[0]).max(),
                   np.abs(pts[:, 1] - center[1]).max()]
    h = 1.12 * max(extent)
    dom_x = [center[0] - h, center[0] + h]
    dom_y = [center[1] - h, center[1] + h]

    def _x(field="A", title="A"):
        return alt.X(f"{field}:Q", title=title,
                     scale=alt.Scale(domain=dom_x, zero=False, nice=False))

    def _y(field="k", title="k"):
        return alt.Y(f"{field}:Q", title=title,
                     scale=alt.Scale(domain=dom_y, zero=False, nice=False))

    ell = ellipse_points(center, C)
    ell_diag = ellipse_points(center, np.diag(np.diag(C)))
    # The `t` column preserves the angular drawing order. Without an explicit
    # order channel, Altair connects line points sorted by x, which turns a
    # closed curve into a zigzag band between its upper and lower branches.
    curves = pd.concat([
        pd.DataFrame({"A": ell[:, 0], "k": ell[:, 1],
                      "t": np.arange(len(ell)), "series": "95% region"}),
        pd.DataFrame({"A": ell_diag[:, 0], "k": ell_diag[:, 1],
                      "t": np.arange(len(ell_diag)),
                      "series": "if uncorrelated"}),
    ])
    # One shared color scale names every mark in the panel, so the legend is
    # the complete explanation and no tooltips are needed. The Monte Carlo
    # entry joins the domain only once a sweep exists; with a fixed domain
    # its legend entry would appear before there was anything to explain.
    domain = ["95% region", "if uncorrelated", "principal axes",
              "estimate", "truth"]
    colors = [COLOR_COV, COLOR_UNCORR, "#475569", "#111827", COLOR_TRUTH]
    # Legend glyph per entry. A merged legend over mixed mark types falls
    # back to a dot for every entry; an explicit shape scale on the same
    # field draws the line series as line strokes.
    symbols = ["stroke", "stroke", "stroke", "circle", "cross"]
    if mc and mc.get("fits"):
        domain.insert(3, "Monte Carlo refits")
        colors.insert(3, COLOR_MC)
        symbols.insert(3, "circle")
    scale = alt.Scale(domain=domain, range=colors)
    shape_scale = alt.Scale(domain=domain, range=symbols)
    # orient="right" puts the legend in its own gutter beside the plot
    # rather than overlaying the plotting area. Kept deliberately compact:
    # together with autosize='pad' below, the legend adds to the svg's total
    # width instead of being carved out of the square plotting area.
    legend = alt.Legend(title=None, orient="right", labelFontSize=13,
                        symbolSize=90, labelLimit=150, offset=8,
                        rowPadding=3)

    layers = []

    layers.append(alt.Chart(curves).mark_line(size=2).encode(
        x=_x(), y=_y(),
        order=alt.Order("t:Q"),
        color=alt.Color("series:N", scale=scale, legend=legend),
        strokeDash=alt.StrokeDash("series:N", legend=None),
    ))

    # Principal axes: the eigenvectors of C, each drawn through the center
    # with half-length sqrt(5.991 * eigenvalue), so their tips touch the 95%
    # region. Rendered truly orthogonal thanks to the square equal-span
    # scales above. Drawn as line marks (two points per axis, split by the
    # detail channel) rather than rules, so the legend glyph is a line.
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 0.0, None)
    ax_rows = []
    for i in range(2):
        tip = evecs[:, i] * np.sqrt(CHI2_95_2DOF * evals[i])
        for sgn in (-1.0, 1.0):
            ax_rows.append({"A": center[0] + sgn * tip[0],
                            "k": center[1] + sgn * tip[1],
                            "axis": i, "series": "principal axes"})
    layers.append(alt.Chart(pd.DataFrame(ax_rows)).mark_line(
        size=1.5, opacity=0.9,
    ).encode(
        x=_x(), y=_y(),
        detail="axis:N",
        color=alt.Color("series:N", scale=scale, legend=legend),
    ))

    if mc and mc.get("fits"):
        pts = np.asarray(mc["fits"], dtype=float)
        layers.insert(0, alt.Chart(
            pd.DataFrame({"A": pts[:, 0], "k": pts[:, 1],
                          "series": "Monte Carlo refits"})
        ).mark_point(filled=True, size=18, opacity=0.35).encode(
            x=_x(), y=_y(),
            color=alt.Color("series:N", scale=scale, legend=legend),
            shape=alt.Shape("series:N", scale=shape_scale, legend=legend),
        ))

    # The truth is the center of everything drawn; the estimate from this one
    # dataset is a single dot, one draw from the same cloud the refits fill in.
    # Filled, like the legend renders its glyphs, so the mark on the plot
    # and the mark in the legend are the same symbol.
    layers.append(alt.Chart(
        pd.DataFrame({"A": [a_true], "k": [k_true], "series": "truth"})
    ).mark_point(size=150, filled=True).encode(
        x=_x(), y=_y(),
        color=alt.Color("series:N", scale=scale, legend=legend),
        shape=alt.Shape("series:N", scale=shape_scale, legend=legend)))
    layers.append(alt.Chart(
        pd.DataFrame({"A": [d["A"]], "k": [d["k"]], "series": "estimate"})
    ).mark_point(size=90, filled=True).encode(
        x=_x(), y=_y(),
        color=alt.Color("series:N", scale=scale, legend=legend),
        shape=alt.Shape("series:N", scale=shape_scale, legend=legend)))

    # Square: with the equal data spans above, this makes pixels-per-unit
    # identical on both axes, so angles (the tilt, the axes' orthogonality)
    # render faithfully. autosize='pad' makes width/height mean the PLOTTING
    # AREA: without it the renderer treats them as the whole svg budget and
    # carves the axes and legend out of the plot, squashing the square.
    return alt.layer(*layers).properties(
        width=PANEL_HEIGHT, height=PANEL_HEIGHT,
        autosize=alt.AutoSizeParams(type="pad", contains="padding"),
    ).configure_axis(**_AXIS)


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
        "Two declarations mark the estimated parameters and the residual "
        "container, and the covariance is then read off the factorization "
        "the solve already produced (the reduced Hessian: that "
        "factorization restricted to the two fitted parameters). There "
        "is no second optimization and "
        "no finite differencing: the cost is one backsolve per fitted "
        "parameter, a triangular solve that reuses the factorization."
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
    C = \hat\sigma^{2}\,\bigl(\text{reduced Hessian}\bigr)^{-1},
    \qquad
    \mathrm{se}(\theta_j) = \sqrt{C_{jj}}
    """)

    st.markdown("#### From the matrix to the ellipse")
    st.markdown(
        "The 95% joint confidence region for the pair is the level set"
    )
    st.latex(r"""
    \Bigl\{\,\theta \;:\;
    (\theta-\theta^{0})^{\mathsf T} C^{-1} (\theta-\theta^{0})
    \;\le\; \chi^{2}_{2,\,0.95} = 5.991 \,\Bigr\}
    """)
    st.markdown(
        "The center $\\theta^{0}$ is the estimate, exactly as a "
        "practitioner would draw it. Every mark in the panel comes from "
        "that one expression:"
    )
    st.latex(r"""
    \text{principal axes:}\quad \sqrt{5.991\,\lambda_i}
    \ \text{along the eigenvectors of } C
    """)
    st.markdown(
        "The diagonal of $C$ alone sets how far the region reaches along "
        "each axis, at $\\pm\\sqrt{5.991}\\,\\mathrm{se} = "
        "\\pm 2.4477\\,\\mathrm{se}$. Zeroing the off-diagonal therefore "
        "produces the faint companion ellipse with exactly the same reach: "
        "the covariance term does not change either parameter's own spread, "
        "it only tilts and reshapes the joint region. A single parameter's "
        "own 95% interval is the shorter $\\pm 1.96\\,\\mathrm{se}$, so "
        "reading the joint region off two individual error bars understates "
        "it by a factor of $2.4477/1.96 = 1.249$ in each direction."
    )

    st.markdown("#### What the Monte Carlo checks")
    st.markdown(
        "The sweep asks a replication question: rerun the experiment, "
        "refit, and see where the new estimate lands relative to the "
        "region you drew. Each draw generates an independent dataset from "
        "the same truth and noise, refits it, and the capture metric "
        "counts the fraction of the refitted estimates inside the drawn "
        "region. The sweep also shows the sample covariance of its "
        "estimates, the same matrix measured by brute force. Expect the "
        "shapes to agree and the scales to differ: the reported matrix "
        "inherits this one dataset's $\\hat\\sigma$, the empirical one "
        "inherits the true noise, and their ratio is roughly "
        "$(\\hat\\sigma/\\sigma)^{2}$. That ratio is fixed by the sampling "
        "seed and does not depend on the noise level: the seed sets the "
        "standardized noise pattern and $\\sigma$ only scales it, so "
        "$\\hat\\sigma/\\sigma$ comes out the same at every $\\sigma$. "
        "Stepping the seed is what moves it."
    )
    st.markdown(
        "Averaged across datasets that fraction is about 78%, not 95%. "
        "The region is centered on your estimate, which carries its own "
        "error, so a replication differs from the center with twice the "
        "covariance. How much of the cloud your particular region "
        "captures depends on where your estimate landed, so the number "
        "moves as you step the sampling seed. The 95% belongs to a "
        "different event: the region contains the **true** parameters for "
        "95% of datasets. On screen that is a single yes-or-no per "
        "dataset, the cross inside the region or not: it lands outside "
        "about one seed in twenty."
    )
    st.markdown(
        "Narrowing the x range to a short late window is worth trying: there "
        "the data pin the local value and slope, while $A$ is a long "
        "extrapolation back to $x=0$. The correlation rises to nearly 1 and "
        "the ellipse collapses to a sliver, which is what an unidentifiable "
        "pair looks like before any prediction is made."
    )

    st.markdown("**References**")
    st.markdown(
        "[1] G. A. F. Seber and C. J. Wild, *Nonlinear Regression*. "
        "Wiley, New York, 1989 (the asymptotic covariance of least-squares "
        "estimates). "
        "[Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/0471725315)"
    )
    st.markdown(
        "[2] G. Cumming and R. Maillardet, \"Confidence intervals and "
        "replication: Where will the next mean fall?\" *Psychological "
        "Methods*, 11(3):217-227, 2006 (the capture-percentage result: a "
        "95% confidence interval captures a replication's estimate well "
        "under 95% of the time). "
        "[DOI](https://doi.org/10.1037/1082-989X.11.3.217)"
    )
    st.markdown(
        "[3] J. Kitchin, *POUNCE*. "
        "[GitHub](https://github.com/jkitchin/pounce); the Pyomo plugin "
        "and `covariance()` ship as "
        "[pyomo-pounce](https://pypi.org/project/pyomo-pounce/)."
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

/* Metric labels (Â, k̂, correlation, σ̂, coverage) run small by default;
   these are the row a reader scans first, so bump them. */
[data-testid="stMetricLabel"] p {
    font-size: 1.1rem !important;
}

/* Metric values sized so "estimate ± se" fits on one line in its column. */
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
}

/* The run-the-sweep note sits in a metric-row slot and must stay on one
   line. Scoped by container key: the other alerts (solver errors, pounce
   notes) are real paragraphs and must keep wrapping. */
.st-key-mc_note [data-testid="stAlert"] p {
    white-space: nowrap;
    font-size: 0.85rem;
}
.st-key-mc_note [data-testid="stAlert"] {
    padding: 0.4rem 0.75rem;
    width: fit-content;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}
</style>
"""


# ── 8. Tabs ──────────────────────────────────────────────────────────────────

def render_main(d, a_true, k_true, sigma):
    if d["status"] != "optimal":
        # Failed fit: the panels stay put. The data panel still has the
        # sample and the true curve, and seeing the truth buried in noise is
        # the visual explanation; the other two slots hold their places so
        # the layout does not jump.
        x = np.asarray(d["x"], dtype=float)
        signal = float(np.mean(np.abs(a_true * np.exp(-k_true * x))))
        snr = signal / sigma if sigma > 0 else np.inf
        left, mid, right, _spacer = st.columns([3.7, 2.05, 5.05, 1.1],
                                               gap="medium")
        with left:
            st.markdown("**Data and fit**")
            st.altair_chart(data_panel(d, a_true, k_true), width="content")
        with mid:
            st.markdown("**Covariance**")
            st.caption("No fit, so there is no covariance to report.")
        with right:
            st.markdown("**Parameter space**")
            st.caption("No fit, so there is no region to draw.")
        if d["status"] == "singular":
            what = ("The fit converged, but there is not enough signal in "
                    "this window to tell A and k apart, so no covariance "
                    "exists.")
        else:
            what = ("There is not enough signal in this window to fit: the "
                    "data are noise around a flat line.")
        st.warning(
            f"{what} Widen the x range toward x = 0, lower the noise, or "
            "add data points. Details on the Logs tab."
        )
        # The numbers behind the message, for the Logs tab.
        d["note_details"] = [
            f"solver status: {d['status']}"
            + (f" ({d['detail']})" if d.get("detail") else ""),
            f"mean |signal| over this window: {signal:.3g}; "
            f"noise sigma: {sigma:g}; signal-to-noise ratio: {snr:.2f}",
        ]
        return

    mc = st.session_state.mc
    C = np.asarray(d["C_plot"], dtype=float)

    # The right column carries the square plot plus its side legend, so it
    # gets the extra width; squeezing the legend into a 5/13 column scales
    # the whole chart down to fit, shrinking the plot with it.
    # Column ratios match the three charts' rendered widths (data ~370,
    # matrix ~170, parameter ~495) with a trailing spacer that soaks up
    # whatever page width is left over. Without the spacer the slack lands
    # INSIDE the row and pushes the panels apart.
    # Ratios match the charts' rendered widths (data ~370, matrix stack
    # ~185 including its rotated title, parameter ~505 with legend), so
    # each column fits its chart exactly and the gaps on both sides of the
    # matrix column are the same uniform column gap.
    left, mid, right, _spacer = st.columns([3.7, 2.05, 5.05, 1.1],
                                           gap="medium")
    with left:
        st.markdown("**Data and fit**")
        st.altair_chart(data_panel(d, a_true, k_true), width="content")
    with mid:
        st.markdown("**Covariance**")
        st.altair_chart(matrix_panel(C, "single fit"), width="content")
        if mc and mc.get("fits") and len(mc["fits"]) > 2:
            # The same matrix measured by brute force: the sample covariance
            # of the refitted estimates. The reported matrix inherits this
            # one dataset's sigma-hat; this one inherits the truth.
            emp = np.cov(np.asarray(mc["fits"], dtype=float).T)
            st.altair_chart(matrix_panel(emp, "Monte Carlo"),
                            width="content")
    with right:
        st.markdown("**Parameter space**")
        # width="content", not "stretch": the panel is a fixed square so
        # that angles render faithfully; stretching would undo that.
        st.altair_chart(parameter_panel(d, a_true, k_true, mc),
                        width="content")

    se = d["se"]
    # The last column is wider: it holds either the coverage metric or the
    # run-the-sweep note, which must fit on one line.
    # The column split differs by state so nothing floats in dead space:
    # with a sweep, six tight slots; without one, the run-the-sweep note
    # sits directly to the right of the last visible number, wide enough
    # that its single line stays inside the box.
    if mc and mc.get("fits"):
        c1, c2, c3, c4, c5, c6 = st.columns([1.3, 1.3, 1.2, 0.9, 1.0, 1.6])
    else:
        c1, c2, c3, c4, c5 = st.columns([1.3, 1.3, 1.2, 0.9, 2.9])
        c6 = c5
    # The ± one-standard-error rides inline with the estimate. True values
    # are not repeated here: they are on the sidebar sliders already.
    c1.metric("Â", f"{d['A']:.4f} ± {se[0]:.4f}")
    c2.metric("k̂", f"{d['k']:.4f} ± {se[1]:.4f}")
    c3.metric("correlation(Â, k̂)", f"{d['corr']:+.3f}")
    c4.metric("σ̂ (single fit)", f"{d['sigma_hat']:.4f}")
    if mc and mc.get("fits"):
        if mc.get("sigma_mc"):
            c5.metric("σ̂ (Monte Carlo)", f"{mc['sigma_mc']:.4f}")
        fits = np.asarray(mc["fits"], dtype=float)
        dif = fits - np.array([d["A"], d["k"]])
        q = np.einsum("ij,jk,ik->i", dif, np.linalg.inv(C), dif)
        cover = float((q <= CHI2_95_2DOF).mean())
        c6.metric("capture of refits", f"{100 * cover:.1f}%")
        if mc["failures"]:
            st.warning(f"{mc['failures']} of {mc['n_draws']} refits did not "
                       "converge and were dropped.")
    else:
        with c5, st.container(key="mc_note"):
            st.info("Run the Monte Carlo sweep in the sidebar", icon="🎲")

    for note in d.get("notes") or []:
        st.warning(note, icon="⚠️")


def render_logs(d):
    log = (d.get("log") or "").strip()
    if log:
        st.caption(
            "POUNCE's output for the displayed fit. The Monte Carlo refits "
            "are solved with logging suppressed; only this fit's log is "
            "kept."
        )
        st.code(log, language="text")
    else:
        st.info("No solver output captured for the last fit.")
    if d.get("note_details"):
        st.caption("diagnostics for this fit, verbatim:")
        st.code("\n".join(d["note_details"]), language="text")


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
    "style='color: #6b7280; text-decoration: underline;'>POUNCE</a>"
    "</span></h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([5, 3])
with _caption_col:
    st.markdown(
        "Fit $y = A\\,e^{-kx}$ to noisy data. The covariance estimate for "
        "$(\\hat{A}, \\hat{k})$ is the noise variance estimate "
        "$\\hat{\\sigma}^{2}$ times the inverse of the reduced Hessian of "
        "the Lagrangian, read off the KKT factorization the solver already "
        "holds: one backsolve per parameter, no second optimization. The "
        "principal axes drawn in parameter space are the "
        "eigenvectors of that matrix, with lengths scaled by the "
        "eigenvalues. The Monte Carlo sweep checks it by brute force, one "
        "full refit per draw."
    )

# ---- Sidebar: the experiment ----
st.sidebar.header("Truth")
st.sidebar.slider("A", 0.5, 5.0, key="a_true", step=0.1, format="%.1f")
st.sidebar.slider("k", 0.2, 3.0, key="k_true", step=0.1, format="%.1f")
st.sidebar.slider("noise σ", 0.005, 0.50, key="sigma", step=0.005,
                  format="%.3f")

# Section header with the dataset seed beside it: stepping the seed IS the
# new-dataset action, so it needs no separate button. The seed column runs
# to the sidebar edge: Streamlit hides the +/- steppers below ~125 px, so
# the column keeps the full remaining width rather than leaving a spacer.
_hd, _seed_col = st.sidebar.columns([2.4, 3.1],
                                    vertical_alignment="bottom")
_hd.header("Sampling")
_seed_col.number_input("seed", 0, 99999, key="seed", step=1)
st.sidebar.slider("x range", 0.0, 6.0, step=0.1, key="x_range",
                  format="%.1f")
st.sidebar.slider("data points", 5, 100, key="n_pts", step=1)

# Header row mirrors the Sampling section: its seed sits beside the title.
# Stepping it selects a fresh block of synthetic datasets and drops the
# stored cloud, so coverage is visibly a sampled number, not a constant.
_mh, _mc_seed_col = st.sidebar.columns([2.4, 3.1],
                                       vertical_alignment="bottom")
_mh.header("Monte Carlo")
_mc_seed_col.number_input("seed", 0, 99999, key="mc_seed", step=1)
st.sidebar.slider("refits", MC_MIN, MC_MAX, key="n_draws", step=MC_STEP,
                  help="Each refit is a full NLP solve. 200 is enough to "
                       "confirm the coverage to within sampling noise.")
run_mc = st.sidebar.button("Run sweep", type="primary", width="stretch")
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
