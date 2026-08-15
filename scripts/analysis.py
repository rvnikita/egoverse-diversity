"""The measurements that make the diversity score falsifiable, plus dashboard inputs.

Everything here runs on the committed 12 MB vector cache: no GPU, no network, no creds.

  1. DUPLICATION TEST     clone part of a set; a valid diversity metric must not rise.
                          Run in two variants — exact clones (the theorem) and clones with
                          noise (the data version a sceptic will ask for).
  2. ATYPICALITY -> FAILURE  a purely label-free score, ranked against the recovered
                          outcome labels. This is the external validity check: the score
                          predicts something it was never fitted to.
  3. ARE FAILURES ATYPICAL?  the mechanism behind (2), tested directly.
  4. PROJECTION           2D PCA coordinates for the dashboard scatter.
  5. CONTACT SHEETS       thumbnails of two concrete subsets, so redundancy is visible
                          rather than asserted.

    .venv/bin/python scripts/analysis.py
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diversity import duplication_test, farthest_point, vendi_score  # noqa: E402

VECTORS = ROOT / "results" / "episode_vectors.npz"
LABELS = ROOT / "out" / "cup_labelled.csv"
CLIPS = ROOT / "out" / "cup_mp4"
RESULTS = ROOT / "results"
SHEETS = RESULTS / "sheets"

K_DEMO = 32          # the two subsets shown side by side on the dashboard
SEED_DEMO = 0


def atypicality(V: np.ndarray, k: int = 20) -> dict[str, np.ndarray]:
    """Label-free "how unlike everything else is this episode" scores."""
    S = V @ V.T
    med = V.mean(axis=0)
    med /= np.linalg.norm(med)
    Soff = S.copy()
    np.fill_diagonal(Soff, -np.inf)
    return {
        "dist_to_centroid": 1 - V @ med,
        "mean_dist_to_all": 1 - (S.sum(1) - 1) / (len(V) - 1),
        f"knn{k}_dist": 1 - np.sort(Soff, axis=1)[:, -k:].mean(1),
    }


def duplication_with_noise(V: np.ndarray, frac: float = 0.3, sigma: float = 0.01,
                           seed: int = 0) -> dict:
    """Near-duplicates, not exact clones: the version that tests DATA, not algebra."""
    rng = np.random.default_rng(seed)
    n = len(V)
    k = max(1, int(n * frac))
    pick = V[rng.choice(n, size=k, replace=False)]
    noisy = pick + rng.normal(0, sigma, pick.shape)
    noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
    before, after = vendi_score(V), vendi_score(np.vstack([V, noisy]))
    # cosine distance actually introduced, so sigma is interpretable next to real data
    dist = float((1 - (pick * noisy).sum(1)).mean())
    return {"n": n, "n_after": n + k, "sigma": sigma,
            "mean_cosine_dist_introduced": round(dist, 5),
            "vendi_before": round(before, 3), "vendi_after": round(after, 3),
            "delta": round(after - before, 3), "passes": bool(after <= before + 1e-6)}


def duplication_sweep(V: np.ndarray, real_nn_dist: float) -> dict:
    """At what perturbation does the score stop being duplication-proof?

    Exact clones provably cannot raise the Vendi Score. Near-duplicates can, because a
    cosine kernel eventually reads a perturbed copy as its own mode. A binary pass/fail
    hides that; the honest answer is the threshold, stated next to how far apart REAL
    near-neighbour episodes actually are.
    """
    rows = []
    for sigma in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
        r = duplication_with_noise(V, sigma=sigma) if sigma else duplication_test(V, frac=0.3)
        rows.append({"sigma": sigma,
                     "cos_dist": r.get("mean_cosine_dist_introduced", 0.0),
                     "delta": r["delta"], "passes": bool(r["passes"])})
    return {"sweep": rows, "real_nn_cosine_dist": round(real_nn_dist, 5)}


def contact_sheet(mp4s: list[pathlib.Path], cols: int = 8, cell: int = 112) -> str:
    """A grid of mid-clip frames as a base64 JPEG, so the dashboard stays self-contained."""
    from PIL import Image

    rows = (len(mp4s) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 20))
    for i, p in enumerate(mp4s):
        try:
            # -sseof seeks from the END: for "did the cup land on the saucer" the final
            # state is the informative frame, and it needs no duration probe.
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-sseof", "-1.5", "-i", str(p), "-frames:v", "1",
                 "-vf", f"scale={cell}:{cell}:force_original_aspect_ratio=increase,"
                        f"crop={cell}:{cell}",
                 "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
                capture_output=True, timeout=20).stdout
            if raw:
                sheet.paste(Image.open(io.BytesIO(raw)), ((i % cols) * cell, (i // cols) * cell))
        except Exception:  # noqa: BLE001 - a missing clip should not kill the dashboard
            continue
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    d = np.load(VECTORS, allow_pickle=False)
    V = d["V_all32"]
    ep_hash = d["episode_hash"].astype(str)
    outcome = d["outcome"].astype(str)
    y = (outcome == "failure")
    n = len(V)

    csv = pd.read_csv(LABELS)
    combo_of = dict(zip(csv.episode_hash.astype(str), csv.objects.astype(str)))
    mp4_of = dict(zip(csv.episode_hash.astype(str),
                      csv.zarr_mp4_path.fillna("").map(lambda u: pathlib.Path(str(u)).name)))
    combo = np.array([combo_of.get(h, "?") for h in ep_hash])

    out: dict = {
        "n_episodes": int(n), "n_failures": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "n_combos": int(len(set(combo.tolist()))),
        "gpu_seconds_total": round(float(d["gpu_seconds"]), 1),
        "ms_per_frame": round(float(d["gpu_seconds"]) / (n * 32) * 1e3, 2),
    }

    # ---------------------------------------------------------------- 1. duplication
    print("== duplication test (a valid diversity metric must not rise) ==")
    sub = V[np.random.default_rng(0).choice(n, 64, replace=False)]
    exact = duplication_test(sub, frac=0.3)
    noisy = duplication_with_noise(sub, frac=0.3, sigma=0.01)
    # how far apart are genuinely distinct REAL episodes? gives sigma a scale.
    Ssub = sub @ sub.T
    np.fill_diagonal(Ssub, -np.inf)
    real_nn = float((1 - Ssub.max(axis=1)).mean())
    out["duplication"] = {"exact_clones": exact, "near_duplicates": noisy,
                          **duplication_sweep(sub, real_nn)}
    for nm, r in (("exact clones", exact), ("near-dupes (sigma=0.01)", noisy)):
        print(f"  {nm:24s} {r['n']}->{r['n_after']} items  "
              f"VS {r['vendi_before']} -> {r['vendi_after']}  passes={r['passes']}")
    print(f"  real nearest-neighbour cosine distance between distinct episodes: {real_nn:.5f}")
    for row in out["duplication"]["sweep"]:
        print(f"    sigma={row['sigma']:<7} injected cos-dist {row['cos_dist']:.5f}  "
              f"delta VS {row['delta']:+.3f}  {'ok' if row['passes'] else 'INFLATES'}")

    # ------------------------------------------------- 2. label-free failure ranking
    print("\n== label-free atypicality as a failure detector ==")
    from sklearn.metrics import roc_auc_score

    rank = {}
    for name, s in atypicality(V).items():
        top = np.argsort(-s)[:K_DEMO]
        rank[name] = {"auroc": round(float(roc_auc_score(y, s)), 4),
                      f"top{K_DEMO}_failures": int(y[top].sum()),
                      f"top{K_DEMO}_enrichment": round(
                          float(y[top].mean() / y.mean()), 2)}
        print(f"  {name:18s} AUROC {rank[name]['auroc']:.3f}   "
              f"top-{K_DEMO}: {rank[name][f'top{K_DEMO}_failures']}/{K_DEMO} failures "
              f"({rank[name][f'top{K_DEMO}_enrichment']}x base)")
    out["label_free_ranking"] = rank

    # ------------------------------------------------- 3. are failures atypical?
    print("\n== are failures intrinsically atypical? ==")
    from scipy.stats import mannwhitneyu

    med = V.mean(axis=0)
    med /= np.linalg.norm(med)
    df_, ds_ = 1 - V[y] @ med, 1 - V[~y] @ med
    rng = np.random.default_rng(0)
    nf = int(y.sum())
    vs = [vendi_score(V[rng.choice(np.where(~y)[0], nf, replace=False)]) for _ in range(200)]
    out["failures_atypical"] = {
        "vendi_failures": round(float(vendi_score(V[y])), 3),
        "vendi_successes_matched_n": round(float(np.mean(vs)), 3),
        "vendi_successes_sd": round(float(np.std(vs)), 3),
        "ratio": round(float(vendi_score(V[y]) / np.mean(vs)), 3),
        "centroid_dist_failures": round(float(df_.mean()), 5),
        "centroid_dist_successes": round(float(ds_.mean()), 5),
        "mannwhitney_p": float(mannwhitneyu(df_, ds_, alternative="greater").pvalue),
    }
    fa = out["failures_atypical"]
    print(f"  Vendi failures {fa['vendi_failures']} vs matched-n successes "
          f"{fa['vendi_successes_matched_n']} ± {fa['vendi_successes_sd']} "
          f"({fa['ratio']}x)")
    print(f"  centroid distance {fa['centroid_dist_failures']} vs "
          f"{fa['centroid_dist_successes']}   Mann-Whitney p={fa['mannwhitney_p']:.2e}")

    # ------------------------------------------------- 3b. metadata coverage
    # The external check on the score. `objects`, `operator` and the recording date are
    # registry metadata the encoder never saw — so if a subset the score calls "more
    # diverse" also covers more of them, the score is tracking something real and not
    # just its own geometry.
    print("\n== metadata coverage: external ground truth the embedding never saw ==")
    from sklearn.cluster import KMeans

    op = d["operator"].astype(str)
    day = np.array([h[:10] for h in ep_hash])
    axes = {"prop combos": combo, "operators": op, "recording days": day}

    def pick(kind, k, s):
        if kind == "random":
            return np.random.default_rng(s).choice(n, k, replace=False)
        if kind == "diverse":
            return farthest_point(V, k, seed=s)
        km = KMeans(n_clusters=k, n_init=3, random_state=s).fit(V)
        return np.unique((V @ km.cluster_centers_.T).argmax(axis=0))

    budgets = [4, 8, 16, 32, 64]
    cov: dict = {a: {"total": int(len(set(v.tolist()))), "budgets": budgets, "by": {}}
                 for a, v in axes.items()}
    for kind in ("random", "diverse"):
        for a, v in axes.items():
            means, sds = [], []
            for k in budgets:
                c = [len(set(v[pick(kind, k, s)].tolist())) for s in range(30)]
                means.append(round(float(np.mean(c)), 2))
                sds.append(round(float(np.std(c)), 2))
            cov[a]["by"][kind] = {"mean": means, "sd": sds}
    out["coverage"] = cov
    for a in axes:
        r_, f_ = cov[a]["by"]["random"]["mean"], cov[a]["by"]["diverse"]["mean"]
        print(f"  {a:16s} (of {cov[a]['total']:2d})  random {r_}  diverse {f_}")

    # ------------------------------------------------- 4. the two demo subsets
    idx_rand = np.random.default_rng(SEED_DEMO).choice(n, K_DEMO, replace=False)
    idx_fps = farthest_point(V, K_DEMO, seed=SEED_DEMO)
    subsets = {}
    for name, idx in (("random", idx_rand), ("diverse", idx_fps)):
        subsets[name] = {
            "k": K_DEMO, "idx": idx.tolist(),
            "vendi": round(float(vendi_score(V[idx])), 3),
            "failures": int(y[idx].sum()),
            "failure_pct": round(float(y[idx].mean()) * 100, 1),
            "combos": int(len(set(combo[idx].tolist()))),
        }
        print(f"\n  subset {name:8s} Vendi {subsets[name]['vendi']:.2f}  "
              f"failures {subsets[name]['failures']}/{K_DEMO}  "
              f"combos {subsets[name]['combos']}/{out['n_combos']}")
    out["subsets"] = subsets

    # ------------------------------------------------- 5. projection + contact sheets
    from sklearn.decomposition import PCA

    P3 = PCA(n_components=3, random_state=0).fit_transform(V)
    P3 = (P3 - P3.mean(0)) / P3.std(0)
    out["projection"] = {"x": np.round(P3[:, 0], 3).tolist(),
                         "y": np.round(P3[:, 1], 3).tolist(),
                         "z": np.round(P3[:, 2], 3).tolist(),
                         "combo": [sorted(set(combo.tolist())).index(c) for c in combo]}

    # ------------------------------------------------- 6. compute-once economics
    # The structural argument: embedding is a one-off index cost, an LLM judge is a
    # per-query cost. Count how many subset scorings this repo's experiment actually did.
    try:
        n_scorings = sum(
            len(pd.read_csv(RESULTS / f"selection_{s}.csv"))
            for s in ("stratified", "session")
            if (RESULTS / f"selection_{s}.csv").exists())
    except Exception:  # noqa: BLE001
        n_scorings = 0
    out["economics"] = {
        "index_gpu_seconds": round(float(d["gpu_seconds"]), 1),
        "index_cost_usd": round(float(d["gpu_seconds"]) / 3600 * 0.80, 4),  # L4 ≈ $0.80/h
        "subset_scorings_performed": int(n_scorings),
        "marginal_cost_per_scoring_usd": 0.0,
        "note": "index built once; every subsequent subset score is numpy on cached vectors",
    }
    print(f"\n== economics ==\n  index: {out['economics']['index_gpu_seconds']}s GPU "
          f"= ${out['economics']['index_cost_usd']:.4f}, then "
          f"{n_scorings:,} subset scorings at zero marginal cost")

    SHEETS.mkdir(parents=True, exist_ok=True)
    if CLIPS.exists():
        print("\n== contact sheets ==")
        for name, idx in (("random", idx_rand), ("diverse", idx_fps)):
            paths = [CLIPS / mp4_of.get(ep_hash[i], "") for i in idx]
            b64 = contact_sheet([p for p in paths if p.exists()])
            (SHEETS / f"{name}.jpg.b64").write_text(b64)
            print(f"  {name}: {len(b64)//1024} KB base64")
    else:
        print(f"\n  !! {CLIPS} absent — contact sheets skipped (dashboard degrades gracefully)")

    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS/'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
