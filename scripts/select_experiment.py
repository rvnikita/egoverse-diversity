"""Does a label-free diversity selector pick a better subset — and better at what?

Reads the cached episode embeddings (no GPU, no network, no credentials) and, for each
selector x budget x seed, records:

  * Vendi score of the subset            -- the diversity score itself
  * failures found                        -- outcome coverage, the label-free payoff
  * prop-combo / operator / day coverage  -- EXTERNAL ground truth from the registry,
                                             which the embedding never saw. This is what
                                             breaks the circularity of "we selected for
                                             spread, then measured spread".
  * balanced accuracy + AUROC             -- the downstream audit

Selectors:
  random          the baseline anyone would use
  fps             farthest-point in embedding space (spread-seeking)
  kmedoid         k-means then nearest real episode to each centroid (cluster-covering;
                  guards against fps merely collecting outliers/corrupt clips)
  matched_random  random, but drawing the SAME number of failures fps realised at this
                  budget+seed. USES LABELS. This is the control that decomposes any
                  accuracy gain into "class balance" vs "everything else". Stratifying to
                  the pool ratio instead would equal plain random in expectation and
                  control for nothing.

Everything about the classifier is frozen a priori and identical across selectors: we are
comparing selectors, not tuning a model. PCA is fit on the POOL only, never on the test
set.

    .venv/bin/python scripts/select_experiment.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diversity import farthest_point, vendi_score  # noqa: E402

# Preferred input is the COMMITTED pooled-vector cache: 12 MB, in the repo, so this whole
# experiment reproduces with no GPU, no AWS credentials and no network. out/emb/cup.npz is
# the 78 MB per-frame tensor the Modal L4 produced; it is not shipped.
VECTORS = ROOT / "results" / "episode_vectors.npz"
FRAME_NPZ = ROOT / "out" / "emb" / "cup.npz"
LABELS = ROOT / "out" / "cup_labelled.csv"
RESULTS = ROOT / "results"

# Frozen classifier hyperparameters. Identical for every selector — the comparison is
# paired, so no feature or hyperparameter choice here can manufacture a difference
# BETWEEN selectors; it can only move all of them together.
PCA_DIM = 16          # n can be as low as 16; components are capped at n_train - 1
LOGREG_C = 1.0
TEST_FRAC = 0.20
SEEDS = 20

# class_weight='balanced' is not optional: the pool is 82/18, and without it logistic
# regression predicts the majority class on small training sets, making balanced accuracy
# exactly 0.5 by construction rather than by measurement.
CLASS_WEIGHT = "balanced"

# The AUDIT feature is the first 4 + last 4 frames, not a mean over all 32. "Did the cup
# end up on the saucer" is a statement about the END STATE; averaging 32 frames dilutes it
# with the approach. Measured on the full pool with 5-fold CV: mean-of-32 AUROC 0.789,
# last-4 0.899, first4+last4 0.906. Chosen from task semantics and reported alongside the
# alternatives, never tuned per selector.
AUDIT_SLOTS_HEAD, AUDIT_SLOTS_TAIL = 4, 4


# ------------------------------------------------------------------ data loading


def load() -> dict:
    if not VECTORS.exists():
        raise SystemExit(
            f"{VECTORS} missing. It ships with the repo; to rebuild it from the raw "
            f"frames run scripts/build_cup_embeddings.py (needs Modal + AWS)."
        )
    d = np.load(VECTORS, allow_pickle=False)
    ep_hash = d["episode_hash"].astype(str)

    # `objects` (the prop combo) is external ground truth the embedding never saw.
    # Join by episode_hash rather than by row order so a filtering change cannot
    # silently misalign the labels.
    csv = pd.read_csv(LABELS)
    combo_of = dict(zip(csv.episode_hash.astype(str), csv.objects.astype(str)))

    return {
        # score/selection spaces, one per frame budget, all pooled on the GPU pass
        "V": {32: d["V_all32"], 8: d["V_8"], 4: d["V_4"]},
        "head4": d["V_head4"], "tail4": d["V_tail4"],
        "n": len(ep_hash), "hash": ep_hash,
        "outcome": d["outcome"].astype(str),
        "operator": d["operator"].astype(str),
        "combo": np.array([combo_of.get(h, "?") for h in ep_hash]),
        # episode_hash is a recording timestamp: YYYY-MM-DD-HH-MM-SS-micros
        "day": np.array([h[:10] for h in ep_hash]),
        "gpu_seconds": float(d["gpu_seconds"]),
    }


def episode_vectors(D: dict, n_frames: int) -> np.ndarray:
    """SELECTION/SCORE vector: mean-pooled DINOv2 over n_frames of the 32 sampled.

    A uniform subsample of a uniform sample is still uniform over the clip, so the whole
    frames-per-episode ablation came out of ONE GPU pass and is precomputed here.
    """
    if n_frames not in D["V"]:
        raise SystemExit(f"no cached vectors for {n_frames} frames; have {sorted(D['V'])}")
    return D["V"][n_frames]


def audit_features(D: dict) -> np.ndarray:
    """AUDIT vector: first 4 and last 4 frames, concatenated. See AUDIT_SLOTS_* above.

    Deliberately a DIFFERENT representation from the one the score and the selectors use.
    If the audit lived in exactly the space the selector maximises, the comparison would
    be circular by construction.
    """
    return np.hstack([D["head4"], D["tail4"]])


# ------------------------------------------------------------------- selectors


def sel_random(V, pool, k, seed, fail, **_):
    return np.random.default_rng(seed).choice(pool, size=k, replace=False)


def sel_fps(V, pool, k, seed, fail, **_):
    return pool[farthest_point(V[pool], k, seed=seed)]


def sel_kmedoid(V, pool, k, seed, fail, **_):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(V[pool])
    # nearest REAL episode to each centroid — a medoid, so the subset stays real data
    sim = V[pool] @ km.cluster_centers_.T          # (n_pool, k)
    idx = np.unique(sim.argmax(axis=0))
    if len(idx) < k:                                # duplicate winners: top up by distance
        extra = np.argsort(-sim.max(axis=1))
        for j in extra:
            if len(idx) >= k:
                break
            if j not in idx:
                idx = np.append(idx, j)
    return pool[idx[:k]]


def sel_matched_random(V, pool, k, seed, fail, n_fail_target=0, **_):
    """Random, but with the failure count fps actually realised. The class-balance control."""
    rng = np.random.default_rng(seed + 10_000)
    pf, ps = pool[fail[pool]], pool[~fail[pool]]
    nf = int(min(n_fail_target, len(pf), k))
    ns = k - nf
    if ns > len(ps):
        ns, nf = len(ps), k - len(ps)
    return np.concatenate([rng.choice(pf, nf, replace=False),
                           rng.choice(ps, ns, replace=False)])


SELECTORS = {"random": sel_random, "fps": sel_fps, "kmedoid": sel_kmedoid,
             "matched_random": sel_matched_random}


# ------------------------------------------------------------------- evaluation


def audit(A, train_idx, test_idx, y, pool) -> tuple[float, float, bool]:
    """Balanced accuracy + AUROC on the audit features. PCA fit on the POOL only."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    ytr = y[train_idx]
    if len(np.unique(ytr)) < 2:
        # Single-class training set: undefined, not skipped — skipping would bias the band.
        return 0.5, 0.5, True

    dim = min(PCA_DIM, len(train_idx) - 1, A.shape[1])
    pca = PCA(n_components=dim, random_state=0).fit(A[pool])
    clf = LogisticRegression(C=LOGREG_C, max_iter=3000, class_weight=CLASS_WEIGHT).fit(
        pca.transform(A[train_idx]), ytr)
    Z = pca.transform(A[test_idx])
    pred, prob = clf.predict(Z), clf.predict_proba(Z)[:, 1]
    return (float(balanced_accuracy_score(y[test_idx], pred)),
            float(roc_auc_score(y[test_idx], prob)), False)


def make_split(D, mode: str, seed: int = 0):  # noqa: D401
    """stratified: random, class-balanced. session: hold out whole recording days.

    The stratified split shares recording sessions across train/test, which inflates
    absolute accuracy (recording day is highly decodable from pixels). The session split
    is the honest one; both are reported.
    """
    n = D["n"]
    y = (D["outcome"] == "failure").astype(int)
    if mode == "session":
        days = np.array(sorted(set(D["day"].tolist())))
        rng = np.random.default_rng(seed)
        rng.shuffle(days)
        test_days, acc = [], 0
        for dy in days:                       # take whole days until ~TEST_FRAC of episodes
            if acc >= TEST_FRAC * n:
                break
            test_days.append(dy)
            acc += int((D["day"] == dy).sum())
        test = np.where(np.isin(D["day"], test_days))[0]
    else:
        rng = np.random.default_rng(seed)
        test = np.concatenate([
            rng.choice(np.where(y == c)[0], int(round(TEST_FRAC * (y == c).sum())),
                       replace=False)
            for c in (0, 1)
        ])
    pool = np.setdiff1d(np.arange(n), test)
    return pool, np.sort(test), y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="16,32,64,128")
    ap.add_argument("--frames", default="4,8,32")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--split", default="stratified", choices=["stratified", "session"])
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    frame_budgets = [int(f) for f in args.frames.split(",")]

    D = load()
    n = D["n"]
    y_fail = (D["outcome"] == "failure")
    n_combo, n_op, n_day = (len(set(D[k].tolist())) for k in ("combo", "operator", "day"))
    print(f"{n} episodes | {y_fail.sum()} failures ({y_fail.mean():.1%}) | "
          f"{n_combo} prop combos | {n_op} operators | {n_day} recording days")

    pool, test, y = make_split(D, args.split)
    print(f"split={args.split}: pool {len(pool)} ({y[pool].sum()} fail) / "
          f"test {len(test)} ({y[test].sum()} fail)\n")

    A = audit_features(D)          # audit space, fixed; never varies by selector
    rows = []
    for nf in frame_budgets:
        V = episode_vectors(D, nf)  # score/selection space
        for k in budgets:
            # fps first: matched_random needs the failure count fps actually drew
            fps_fail = {}
            for name in ("fps", "kmedoid", "random", "matched_random"):
                for seed in range(args.seeds):
                    if name == "kmedoid" and seed >= 5:
                        continue          # deterministic enough; 5 inits is plenty
                    idx = SELECTORS[name](V, pool, k, seed, y_fail,
                                          n_fail_target=fps_fail.get(seed, 0))
                    if name == "fps":
                        fps_fail[seed] = int(y_fail[idx].sum())
                    ba, auc, degen = audit(A, idx, test, y, pool)
                    rows.append({
                        "frames": nf, "k": k, "selector": name, "seed": seed,
                        "vendi": vendi_score(V[idx]),
                        "n_fail": int(y_fail[idx].sum()),
                        "combo_cov": len(set(D["combo"][idx].tolist())),
                        "op_cov": len(set(D["operator"][idx].tolist())),
                        "day_cov": len(set(D["day"][idx].tolist())),
                        "bal_acc": ba, "auroc": auc, "degenerate": degen,
                    })
            print(f"  frames={nf:2d} k={k:3d} done", flush=True)

    res = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / f"selection_{args.split}.csv", index=False)

    # ---------------------------------------------------------------- report
    main_f = max(frame_budgets)
    sub = res[res.frames == main_f]
    print(f"\n=== selectors @ {main_f} frames/episode, split={args.split} ===")
    agg = sub.groupby(["k", "selector"]).agg(
        vendi=("vendi", "mean"), fails=("n_fail", "mean"), fails_sd=("n_fail", "std"),
        combo=("combo_cov", "mean"), op=("op_cov", "mean"),
        bal_acc=("bal_acc", "mean"), auroc=("auroc", "mean"),
    ).round(3)
    print(agg.to_string())

    print(f"\n=== frames/episode ablation (is the frame knob free?) k=32 ===")
    ab = res[res.k == 32].groupby(["frames", "selector"]).agg(
        vendi=("vendi", "mean"), fails=("n_fail", "mean"), auroc=("auroc", "mean")
    ).round(3)
    print(ab.to_string())

    print("\n=== Vendi vs AUROC, WITHIN each budget (pooling across budgets would just "
          "correlate size with size) ===")
    from scipy.stats import spearmanr  # noqa: PLC0415

    for k in budgets:
        s = sub[sub.k == k]
        rho, p = spearmanr(s.vendi, s.auroc)
        print(f"  k={k:3d}  rho={rho:+.3f}  p={p:.3f}  (n={len(s)})")

    # headline numbers, for the dashboard and the slide
    hl = {}
    for k in budgets:
        s = sub[sub.k == k]
        g = {sel: s[s.selector == sel] for sel in SELECTORS}
        hl[k] = {
            sel: {
                "vendi": round(g[sel].vendi.mean(), 3), "vendi_sd": round(g[sel].vendi.std(), 3),
                "fails": round(g[sel].n_fail.mean(), 2), "fails_sd": round(g[sel].n_fail.std(), 2),
                "combo_cov": round(g[sel].combo_cov.mean(), 2),
                "bal_acc": round(g[sel].bal_acc.mean(), 4),
                "auroc": round(g[sel].auroc.mean(), 4),
            } for sel in SELECTORS
        }
        hl[k]["enrichment_fps_over_base"] = round(
            g["fps"].n_fail.mean() / (y_fail.mean() * k), 2)
    (RESULTS / f"headline_{args.split}.json").write_text(json.dumps({
        "n_episodes": int(n), "n_failures": int(y_fail.sum()),
        "base_rate": round(float(y_fail.mean()), 4),
        "n_combos": n_combo, "n_operators": n_op, "n_days": n_day,
        "split": args.split, "pool": len(pool), "test": len(test),
        "frames_per_episode": main_f, "by_budget": hl,
    }, indent=2))
    print(f"\nwrote {RESULTS/f'selection_{args.split}.csv'} and "
          f"{RESULTS/f'headline_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
