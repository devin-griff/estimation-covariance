# Estimation Covariance

Parameter uncertainty from one solve, checked against Monte Carlo

**Live demo:** https://estimation-covariance.griffith-pse.com  
**Home:** https://griffith-pse.com

## What it shows

Fit `y = A·exp(-k·x)` to noisy data and ask the solver how certain the two
fitted parameters are. The uncertainty comes out of the *same* solve:
`declare_fitted` marks the estimated parameters, `declare_residual` marks the
residual container, and [pounce](https://github.com/jkitchin/pounce)'s
`covariance()` then reads the parameter covariance off the KKT factorization
the solve already produced. One backsolve per parameter, no second
optimization.

Three panels, left to right, are one object seen three ways:

1. **Data and fit**, in x and y.
2. **The covariance matrix**, colored by the role each entry plays on the right.
3. **Parameter space**, where that matrix becomes a 95% confidence region.

The third panel draws the matrix rather than illustrating it. The diagonal
entries alone fix the dashed bounding box, at `sqrt(5.991)` standard errors.
Zeroing the off-diagonal gives the faint companion ellipse, which is inscribed
in the *same* box: the covariance term does not change either parameter's own
spread, it only tilts and squeezes the joint region inside it. The crosshair is
each parameter's marginal 95% interval, shorter than the box by a factor of
1.249, which is why a joint region cannot be read off two individual error
bars.

The Monte Carlo sweep refits hundreds of independent synthetic datasets and
counts how many land inside the region. It exists to make the cost visible:
the ellipse takes a few milliseconds, and reproducing it by brute force takes
one full NLP solve per draw.

Narrowing the x range to a short late window drives the correlation to nearly
1 and collapses the ellipse to a sliver, which is what an unidentifiable
parameter pair looks like before any prediction is made.

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

The solver ships as a pip wheel (`pyomo-pounce`) with its binary bundled, so
there is nothing else to install and no license to configure.

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## References

- G. A. F. Seber and C. J. Wild, *Nonlinear Regression*. Wiley, New York,
  1989. The asymptotic covariance of least-squares estimates.
  [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/0471725315)
- G. Cumming and R. Maillardet, "Confidence intervals and replication:
  Where will the next mean fall?" *Psychological Methods*, 11(3):217-227,
  2006. The capture-percentage result: a 95% confidence interval captures
  a replication's estimate well under 95% of the time.
  [DOI](https://doi.org/10.1037/1082-989X.11.3.217)
- J. Kitchin, POUNCE: [github.com/jkitchin/pounce](https://github.com/jkitchin/pounce);
  the Pyomo plugin and `covariance()` ship as
  [pyomo-pounce](https://pypi.org/project/pyomo-pounce/).

## Files

- `app.py`: Streamlit UI and computation
- `requirements.txt`: Python deps
- `favicon.png`: Griffith PSE blackletter G favicon
- `Dockerfile`, `fly.toml`, `.dockerignore`: Fly.io production image config
- `.streamlit/config.toml`: Streamlit defaults for local development (the
  image does not copy it; production uses the flags in the `Dockerfile` CMD)
- `.github/workflows/deploy.yml`: auto-deploy pipeline
