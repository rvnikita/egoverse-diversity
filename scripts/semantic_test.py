"""Does the embedding space agree with the organizer's stated intuition?

He said: folding a t-shirt and pulling a piece of cloth are CLOSE (same objects, similar
manipulation); folding a t-shirt and the dishwasher are FAR. That is a claim about
semantic distance between manipulations, and it is testable.

The registry gives us near-free test pairs because the taxonomy has decayed:
  iron_clothes  vs ironing_clothes  -> literally the same activity, two names  (must be closest)
  fold_clothes  vs fold_laundry     -> same activity, two names                (must be very close)
  fold_*        vs wash_dishes      -> different activity                      (must be far)

Run against each pooling strategy, because how frames become an episode vector decides
whether the score measures the room or the manipulation.

    .venv/bin/python scripts/semantic_test.py
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diversity import POOLERS, vendi_score  # noqa: E402

NPZ = ROOT / "out" / "emb" / "semantic.npz"

# Pairs whose ordering we assert, from the organizer's intuition + the taxonomy.
SYNONYM_PAIRS = [("iron_clothes", "ironing_clothes"), ("fold_clothes", "fold_laundry")]
DISTANT_PAIRS = [("fold_clothes", "wash_dishes"), ("fold_laundry", "wash_dishes"),
                 ("iron_clothes", "wash_dishes")]


def episode_vectors(X, owner, pooler):
    """Collapse per-frame embeddings into one vector per episode."""
    vecs, ids = [], []
    for ep in np.unique(owner):
        F = X[owner == ep]
        if F.shape[0] < 2:
            continue
        vecs.append(pooler(F))
        ids.append(int(ep))
    return np.vstack(vecs), np.array(ids)


def main() -> int:
    d = np.load(NPZ, allow_pickle=False)
    X, owner, task = d["X"], d["owner"], d["task"]
    print(f"loaded {X.shape[0]} frames / {len(np.unique(owner))} episodes / "
          f"{len(set(task.tolist()))} tasks\n")

    results = {}
    for pname, pooler in POOLERS.items():
        V, ids = episode_vectors(X, owner, pooler)
        tasks_of = task[ids]
        names = sorted(set(tasks_of.tolist()))

        # Mean cosine similarity between the episode vectors of each task pair.
        cent = {}
        for t in names:
            cent[t] = V[tasks_of == t]
        sim = {}
        for a, b in itertools.combinations_with_replacement(names, 2):
            A, B = cent[a], cent[b]
            s = float((A @ B.T).mean())
            sim[(a, b)] = sim[(b, a)] = s

        print(f"=== pooling: {pname} ===")
        w = max(len(n) for n in names) + 1
        print(" " * w + "".join(f"{n[:11]:>12}" for n in names))
        for a in names:
            print(f"{a:<{w}}" + "".join(f"{sim[(a,b)]:12.3f}" for b in names))

        syn = [sim[p] for p in SYNONYM_PAIRS]
        dist = [sim[p] for p in DISTANT_PAIRS]
        gap = float(np.mean(syn) - np.mean(dist))
        ok = min(syn) > max(dist)
        print(f"  synonym pairs  mean={np.mean(syn):.3f}  (min {min(syn):.3f})")
        print(f"  distant pairs  mean={np.mean(dist):.3f}  (max {max(dist):.3f})")
        print(f"  separation gap = {gap:+.3f}   ordering holds: {ok}\n")
        results[pname] = {"gap": gap, "ok": ok, "sim": sim, "V": V, "tasks": tasks_of}

    best = max(results, key=lambda k: results[k]["gap"])
    print(f"BEST POOLING: {best}  (gap {results[best]['gap']:+.3f})")

    print("\n=== per-task diversity (Vendi), best pooling ===")
    V, tasks_of = results[best]["V"], results[best]["tasks"]
    for t in sorted(set(tasks_of.tolist())):
        sub = V[tasks_of == t]
        print(f"  {t:18} n={len(sub):3d}  VS={vendi_score(sub):6.2f}")
    print(f"  {'ALL COMBINED':18} n={len(V):3d}  VS={vendi_score(V):6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
