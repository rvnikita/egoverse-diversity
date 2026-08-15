"""The baseline the track names as the problem: an LLM-as-a-judge scoring diversity.

Same two subsets, same question, head to head against the embedding score on the three
things that matter — does it rank them the same way, what does it cost, and does it give
the same answer twice.

Also runs the duplication test ON THE JUDGE: a grid where 30% of the clips are exact
copies of other clips in the same grid must score LOWER on diversity. That is the same
falsification we hold our own score to.

    OPENAI_API_KEY=... .venv/bin/python scripts/llm_judge.py

Costs roughly $0.05 and ~13 API calls. Writes results/llm_judge.json.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import statistics
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SHEETS = RESULTS / "sheets"

MODEL = "gpt-4o"
REPEATS = 5
# OpenAI list prices, USD per 1M tokens, as of the build date. Stated, not measured.
PRICE_IN, PRICE_OUT = 2.50, 10.00

PROMPT = (
    "This image is a grid of frames, one frame taken from each of 32 robot manipulation "
    "video episodes of the same task (placing a cup on a saucer).\n\n"
    "Rate the DIVERSITY of this set of 32 episodes on a scale from 0 to 100, where 0 means "
    "all 32 episodes are essentially identical and 100 means they are maximally varied "
    "(different objects, placements, outcomes, viewpoints).\n\n"
    "Reply with strict JSON only: {\"diversity\": <number 0-100>, \"reason\": \"<one short "
    "sentence>\"}"
)


def call(img_b64: str, temperature: float | None = None) -> tuple[dict, float, dict]:
    body = json.dumps({
        "model": MODEL,
        **({"temperature": temperature} if temperature is not None else {}),
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}},
        ]}],
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.load(r)
    dt = time.monotonic() - t0
    return json.loads(payload["choices"][0]["message"]["content"]), dt, payload["usage"]


def duplicated_sheet(b64: str, frac: float = 0.3, cols: int = 8, cell: int = 112) -> str:
    """Overwrite `frac` of the grid cells with copies of other cells in the same grid."""
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    rows = img.height // cell
    n = rows * cols
    k = int(n * frac)
    for i in range(k):
        src = ((i * 3) % n)
        dst = n - 1 - i
        box = ((src % cols) * cell, (src // cols) * cell,
               (src % cols) * cell + cell, (src // cols) * cell + cell)
        img.paste(img.crop(box), ((dst % cols) * cell, (dst // cols) * cell))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def score_many(name: str, b64: str, repeats: int, temperature: float | None = None) -> dict:
    scores, lat, cost = [], [], 0.0
    for i in range(repeats):
        try:
            out, dt, us = call(b64, temperature)
        except urllib.error.HTTPError as e:  # noqa: PERF203
            print(f"  {name} run {i+1}: HTTP {e.code} {e.read()[:200]!r}")
            continue
        scores.append(float(out["diversity"]))
        lat.append(dt)
        cost += us["prompt_tokens"] / 1e6 * PRICE_IN + us["completion_tokens"] / 1e6 * PRICE_OUT
        print(f"  {name} run {i+1}/{repeats}: {out['diversity']:>5}  ({dt:.1f}s)  "
              f"{out.get('reason','')[:70]}")
    if not scores:
        return {"error": "all calls failed"}
    return {
        "scores": scores,
        "mean": round(statistics.mean(scores), 2),
        "sd": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
        "spread": round(max(scores) - min(scores), 2),
        "latency_s_mean": round(statistics.mean(lat), 2),
        "cost_usd": round(cost, 5),
    }


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping. The dashboard degrades gracefully.")
        return 0
    sheets = {n: (SHEETS / f"{n}.jpg.b64").read_text() for n in ("random", "diverse")
              if (SHEETS / f"{n}.jpg.b64").exists()}
    if len(sheets) < 2:
        print("contact sheets missing — run scripts/analysis.py first")
        return 1

    ours = json.loads((RESULTS / "analysis.json").read_text())["subsets"]
    out: dict = {"model": MODEL, "repeats": REPEATS, "prompt": PROMPT,
                 "price_per_1m_in_out_usd": [PRICE_IN, PRICE_OUT]}

    print(f"== {MODEL} scoring the same two subsets, {REPEATS}x each ==")
    for name in ("random", "diverse"):
        out[name] = score_many(name, sheets[name], REPEATS)

    # The obvious rebuttal is "you left temperature at its default". Close it off.
    print(f"\n== same two subsets at temperature=0 ==")
    for name in ("random", "diverse"):
        out[f"{name}_temp0"] = score_many(f"{name}@T0", sheets[name], REPEATS, temperature=0.0)

    print("\n== duplication test, run on the judge ==")
    dup_b64 = duplicated_sheet(sheets["diverse"])
    out["diverse_30pct_duplicated"] = score_many("dup", dup_b64, 3)

    r, d = out["random"], out["diverse"]
    dup = out["diverse_30pct_duplicated"]
    if "mean" in r and "mean" in d:
        out["agrees_with_us"] = bool(
            (d["mean"] > r["mean"]) ==
            (ours["diverse"]["vendi"] > ours["random"]["vendi"]))
        out["total_cost_usd"] = round(
            r["cost_usd"] + d["cost_usd"] + dup.get("cost_usd", 0), 5)
        out["cost_per_subset_usd"] = round((r["cost_usd"] + d["cost_usd"]) / 2 / REPEATS, 5)
        out["dup_test_passes"] = bool(dup.get("mean", 1e9) <= d["mean"])

        print(f"\n  {MODEL:8s} random {r['mean']:.1f} (sd {r['sd']}, spread {r['spread']})"
              f"   diverse {d['mean']:.1f} (sd {d['sd']}, spread {d['spread']})")
        print(f"  ours     random {ours['random']['vendi']:.2f} (sd 0 — deterministic)"
              f"   diverse {ours['diverse']['vendi']:.2f} (sd 0)")
        print(f"  same ranking: {out['agrees_with_us']}")
        print(f"  duplication test on the judge: {d['mean']:.1f} -> {dup.get('mean')} "
              f"passes={out['dup_test_passes']}")
        r0, d0 = out.get("random_temp0", {}), out.get("diverse_temp0", {})
        if "mean" in r0 and "mean" in d0:
            out["agrees_with_us_temp0"] = bool(d0["mean"] > r0["mean"])
            print(f"  at T=0:  random {r0['mean']:.1f} (sd {r0['sd']}, spread {r0['spread']})"
                  f"   diverse {d0['mean']:.1f} (sd {d0['sd']}, spread {d0['spread']})"
                  f"   same ranking: {out['agrees_with_us_temp0']}")
        print(f"  cost ${out['total_cost_usd']:.4f}, {r['latency_s_mean']:.1f}s per call")

    (RESULTS / "llm_judge.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS/'llm_judge.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
