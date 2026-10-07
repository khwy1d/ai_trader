import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def bootstrap_mean_ci(r, n_boot=5000, alpha=0.05, seed=0):
    r = np.asarray(r, float)
    boots = np.random.default_rng(seed).choice(r, size=(n_boot, len(r))).mean(axis=1)
    return r.mean(), tuple(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def expected_max_normal(n_trials: int) -> float:
    """Approximate E[max of N independent standard normals] (extreme-value approximation). Use: the best of N noise-only
    experiments is expected to look this many standard errors above zero."""
    if n_trials < 2:
        return 0.0
    return float((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e)))
