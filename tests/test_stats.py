import math

import numpy as np

from dap.stats import (
    binned_rate,
    binom_two_sided,
    bootstrap_mean_ci,
    holm,
    mcnemar_exact,
    paired_bootstrap,
    wilson,
)


def test_wilson_known_values():
    # Brown, Cai & DasGupta (2001)-style check: 0/10 has a nonzero upper bound.
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    assert all(math.isnan(v) for v in wilson(0, 0))
    for n in (1, 7, 30, 150):  # exact endpoints
        assert wilson(0, n)[0] == 0.0 and wilson(n, n)[1] == 1.0


def test_exact_binomial_and_mcnemar():
    assert binom_two_sided(0, 0) == 1.0
    assert abs(binom_two_sided(0, 5) - 2 / 32) < 1e-12
    assert abs(binom_two_sided(5, 5) - 2 / 32) < 1e-12
    assert binom_two_sided(3, 6) == 1.0
    a = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0], bool)
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1], bool)
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (5, 1)
    assert abs(r["p"] - binom_two_sided(1, 6)) < 1e-12  # = 14/64
    assert abs(r["p"] - 14 / 64) < 1e-12


def test_bootstrap_cis_cover_and_pair():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 1.0, 400)
    m, lo, hi = bootstrap_mean_ci(x, n_boot=2000)
    assert lo < m < hi and lo < 1.0 < hi
    # Paired: a shared per-seed effect cancels, so the difference is tight.
    base = rng.normal(0, 5, 300)
    r = paired_bootstrap(base + 0.5 + rng.normal(0, 0.1, 300), base, n_boot=2000)
    assert r["n"] == 300 and r["lo"] < 0.5 < r["hi"] and r["hi"] - r["lo"] < 0.1
    assert r["p_boot"] < 0.01
    # NaN pairs are dropped, not imputed.
    r = paired_bootstrap([1.0, np.nan, 3.0], [0.0, 1.0, np.nan], n_boot=100)
    assert r["n"] == 1 and r["mean_diff"] == 1.0


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_binned_rate_includes_the_last_edge():
    rows = binned_rate([0.1, 0.2, 0.5, 0.9], [1, 0, 1, 1], [0.0, 0.5, 0.9])
    assert [r["n"] for r in rows] == [2, 2]
    assert [r["k"] for r in rows] == [1, 2]
