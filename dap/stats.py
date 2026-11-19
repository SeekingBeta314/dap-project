"""Statistics for paired trial results (no SciPy dependency).

Design: every condition runs on the same seeds, and a seed fixes the
scenario, so conditions are compared PAIRED, seed by seed.
  - Success rates: Wilson score intervals from pooled counts (Wilson 1927;
    recommended over the Wald interval by Brown, Cai & DasGupta 2001).
  - Paired success between two conditions: exact McNemar test, i.e. a
    two-sided binomial test on the discordant pairs (McNemar 1947).
  - Paired continuous outcomes (e.g. the loss): percentile bootstrap of the
    mean per-seed difference, resampling seeds (Efron & Tibshirani 1993).
  - Several tests in one analysis: Holm's step-down correction (Holm 1979).
  - Success against a continuous covariate (partner speed): counts and
    Wilson intervals per bin.
"""

import math

import numpy as np

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes out of n (nan, nan if n == 0)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    # Exact endpoints at k = 0 and k = n (floating point lands a hair off).
    lo = 0.0 if k == 0 else max(0.0, c - h)
    hi = 1.0 if k == n else min(1.0, c + h)
    return lo, hi


def binom_two_sided(k, n):
    """Exact two-sided p-value of k successes in n fair coin flips
    (doubling the smaller tail, capped at 1)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_exact(a, b):
    """Paired binary outcomes a, b (same seeds, same order). Returns the
    counts of discordant pairs and the exact two-sided p-value."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired")
    n10 = int(np.sum(a & ~b))  # a succeeded, b failed
    n01 = int(np.sum(~a & b))
    return {"n": int(a.size), "a_only": n10, "b_only": n01,
            "diff": (n10 - n01) / a.size if a.size else float("nan"),
            "p": binom_two_sided(n10, n10 + n01)}


def bootstrap_mean_ci(x, n_boot=10000, alpha=0.05, rng=None):
    """Mean of x and its percentile-bootstrap CI (resampling entries)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, n_boot=10000, alpha=0.05, rng=None):
    """Mean of the paired differences a - b (seeds where both are finite)
    with a percentile-bootstrap CI over seeds, and a two-sided bootstrap
    p-value for "mean difference = 0": twice the smaller share of resampled
    means on either side of 0, with the +1 correction so it is never 0
    (its floor is about 2 / n_boot)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired")
    d = (a - b)[np.isfinite(a) & np.isfinite(b)]
    if d.size == 0:
        return {"n": 0, "mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_boot": float("nan")}
    rng = rng if rng is not None else np.random.default_rng(0)
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    tail = min(np.sum(means <= 0), np.sum(means >= 0))
    p = 2 * (tail + 1) / (n_boot + 1)
    return {"n": int(d.size), "mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_boot": float(min(1.0, p))}


def holm(pvals):
    """Holm step-down adjusted p-values (same order as given). NaN entries
    (tests that could not be run) stay NaN and do not count toward m."""
    p = np.asarray(pvals, float)
    adj = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bin_index(x, edges):
    """Bin of each x for bins [e0, e1), [e1, e2), ..., [e_{n-1}, e_n] (the
    last bin is closed). -1 for values outside [e0, e_n] or NaN."""
    x = np.asarray(x, float)
    edges = np.asarray(edges, float)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = len(edges) - 2
    idx[(x < edges[0]) | (x > edges[-1]) | ~np.isfinite(x)] = -1
    return idx


def binned_rate(x, success, edges):
    """Success counts and Wilson intervals of `success` per bin of `x`
    (bins as in `bin_index`). Returns a list of dicts (bin_lo, bin_hi, n, k,
    rate, ci_lo, ci_hi). Values outside the edges are in no bin: widen the
    edges (see `covering_edges`) rather than lose them."""
    success = np.asarray(success, bool)
    idx = bin_index(x, edges)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = idx == i
        n, k = int(m.sum()), int(success[m].sum())
        c_lo, c_hi = wilson(k, n)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": n, "k": k,
                     "rate": k / n if n else float("nan"), "ci_lo": c_lo, "ci_hi": c_hi})
    return rows


def covering_edges(edges, x):
    """`edges`, with the first and last moved out to include every finite x."""
    edges = [float(e) for e in edges]
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size:
        edges[0] = min(edges[0], float(x.min()))
        edges[-1] = max(edges[-1], float(x.max()))
    return edges
