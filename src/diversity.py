"""Diversity scoring over embeddings — the metric layer.

The headline metric is the **Vendi Score** (Friedman & Dieng, 2023): the exponential of
the Shannon entropy of the eigenvalues of the normalised similarity matrix. Chosen over
the alternatives for three reasons that hold up under questioning:

  * **Interpretable units.** VS = 12.4 means "this set has the diversity of ~12
    effectively distinct items". A raw mean-pairwise-distance has no such reading.
  * **No reference distribution.** Unlike FID, nothing has to be assumed about a
    "true" distribution — important here, where no ground-truth data distribution exists.
  * **Duplication-proof by construction.** Cloning items cannot raise the score. That
    gives a decisive, falsifiable test rather than an argument.

Also provided: coverage/k-center radius as a second, differently-shaped view, and
farthest-point sampling used both for within-episode frame selection and for
across-episode subset selection — the same operation at two scales.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ core metric


def vendi_score(X: np.ndarray, q: float = 1.0, normalize: bool = True) -> float:
    """Effective number of distinct items in `X` (n, d) under a cosine kernel.

    q=1 is the Shannon/exponential-entropy case (the standard Vendi Score).
    """
    if X.shape[0] == 0:
        return 0.0
    if X.shape[0] == 1:
        return 1.0
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    n = X.shape[0]
    # K/n shares eigenvalues with the Gram matrix scaled by 1/n; use whichever is
    # smaller so this stays fast when d << n.
    K = (X @ X.T) / n
    vals = np.linalg.eigvalsh(K)
    vals = np.clip(vals.real, 0, None)
    s = vals.sum()
    if s <= 0:
        return 0.0
    p = vals / s
    p = p[p > 1e-12]
    if q == 1.0:
        return float(np.exp(-(p * np.log(p)).sum()))
    if q == np.inf:
        return float(1.0 / p.max())
    return float((p**q).sum() ** (1.0 / (1.0 - q)))


def coverage_radius(X: np.ndarray, k: int | None = None) -> float:
    """Mean distance from each item to its nearest neighbour — a spread view.

    Complements Vendi: Vendi asks "how many distinct modes", this asks "how spread out".
    A set can score high on one and low on the other, which is diagnostic.
    """
    if X.shape[0] < 2:
        return 0.0
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.max(axis=1)
    return float((1.0 - nn).mean())


# --------------------------------------------------------------- selection ops


def farthest_point(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Greedy k-center. Indices of k items that maximally cover the set.

    Used at two scales: picking representative FRAMES inside one episode, and picking
    representative EPISODES inside a subset. Same operation, same justification.
    """
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(n))]
    d = 1.0 - X @ X[picked[0]]
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, 1.0 - X @ X[nxt])
    return np.array(picked)


# ------------------------------------------------------- episode-level pooling
#
# How frames become one vector per episode. This is NOT a preprocessing detail: it
# decides which axis the diversity score ends up measuring. Mean-pooling uniform
# frames is dominated by whatever is constant across the clip — i.e. the room — so a
# score built on it largely measures backgrounds. Keyframe pooling emphasises the
# frames where the scene is actually changing, i.e. the manipulation.


def pool_mean(F: np.ndarray) -> np.ndarray:
    v = F.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def pool_keyframes(F: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean of the k most mutually-distinct frames (farthest-point within the episode)."""
    if F.shape[0] <= k:
        return pool_mean(F)
    return pool_mean(F[farthest_point(F, k)])


def pool_delta(F: np.ndarray) -> np.ndarray:
    """Mean of consecutive frame differences — what CHANGED, with the static scene
    subtracted out. The most aggressive way to strip background from the descriptor."""
    if F.shape[0] < 2:
        return pool_mean(F)
    d = np.diff(F, axis=0)
    v = d.mean(axis=0)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-8 else pool_mean(F)


POOLERS = {"mean": pool_mean, "keyframe": pool_keyframes, "delta": pool_delta}


# ------------------------------------------------------------------ diagnostics


def duplication_test(X: np.ndarray, frac: float = 0.3, seed: int = 0) -> dict:
    """Clone `frac` of the set and re-score. A valid diversity metric must not rise.

    This is the decisive falsification test: it needs no ground truth and no opinion.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, int(n * frac))
    dup = X[rng.choice(n, size=k, replace=False)]
    before = vendi_score(X)
    after = vendi_score(np.vstack([X, dup]))
    return {
        "n": n,
        "n_after": n + k,
        "vendi_before": round(before, 3),
        "vendi_after": round(after, 3),
        "delta": round(after - before, 3),
        # Items grew by k but distinct modes should not; per-item diversity must fall.
        "passes": after <= before + 1e-6,
    }
