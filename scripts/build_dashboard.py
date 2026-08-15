"""Render the Track 2 dashboard: score diversity once, query it forever.

Self-contained HTML — inline SVG, one small hand-written canvas renderer, inline base64
thumbnails. No external requests, no chart libraries. Reads only committed artefacts.

    .venv/bin/python scripts/build_dashboard.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import slide_parts as SP  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "index.html"   # one file: the slide, then the evidence under it

# Validated 3-slot categorical palette (all-pairs, light and dark) — see dataviz skill.
C = {
    "pool_l": "#a8a7a1", "pool_d": "#5c5b56",
    "s1_l": "#2a78d6", "s1_d": "#3987e5",     # random
    "s2_l": "#eb6834", "s2_d": "#d95926",     # diverse
    "s3_l": "#1baf7a", "s3_d": "#199e70",     # ours, in comparisons
    "neg_l": "#e34948", "neg_d": "#e66767",
}

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  color-scheme:light;
  --bg:#f7f7f5; --surface:#fcfcfb; --line:#e3e2dd;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#7c7b76;
  --pool:%(pool_l)s; --s1:%(s1_l)s; --s2:%(s2_l)s; --s3:%(s3_l)s; --neg:%(neg_l)s;
  --grid:#eceae5;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --bg:#141413; --surface:#1a1a19; --line:#33322e;
  --ink:#ffffff; --ink2:#c3c2b7; --ink3:#8f8e86;
  --pool:%(pool_d)s; --s1:%(s1_d)s; --s2:%(s2_d)s; --s3:%(s3_d)s; --neg:%(neg_d)s;
  --grid:#26251f;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#141413; --surface:#1a1a19; --line:#33322e;
  --ink:#ffffff; --ink2:#c3c2b7; --ink3:#8f8e86;
  --pool:%(pool_d)s; --s1:%(s1_d)s; --s2:%(s2_d)s; --s3:%(s3_d)s; --neg:%(neg_d)s;
  --grid:#26251f;
}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1100px;margin:0 auto;padding:44px 24px 80px}
h1{font-size:34px;line-height:1.15;margin:0 0 10px;letter-spacing:-.025em}
.sub{color:var(--ink2);margin:0;font-size:17px;max-width:62ch}
.meta{color:var(--ink3);font-size:13px;margin:16px 0 0}
h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);
  margin:52px 0 14px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media(max-width:860px){.grid2,.grid3{grid-template-columns:1fr}}
.hero{font-size:46px;font-weight:650;letter-spacing:-.03em;line-height:1}
.hero.sm{font-size:34px}
.herol{font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.07em;
  margin-bottom:8px;font-weight:600}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%%;margin-right:7px;
  vertical-align:middle}
.kv{display:flex;justify-content:space-between;padding:7px 0;border-top:1px solid var(--line);
  font-size:14px}
.kv span:last-child{font-variant-numeric:tabular-nums;font-weight:550}
.sheet{width:100%%;border-radius:6px;display:block;margin-top:12px;border:1px solid var(--line)}
table{width:100%%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-weight:600;color:var(--ink3);font-size:11px;letter-spacing:.06em;
  text-transform:uppercase;padding:0 10px 8px 0;border-bottom:1px solid var(--line)}
td{padding:10px 10px 10px 0;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
td:first-child{font-variant-numeric:normal;color:var(--ink2)}
.win{color:var(--s3);font-weight:650}
.lose{color:var(--neg);font-weight:650}
.note{color:var(--ink2);font-size:13.5px;line-height:1.6;margin:12px 0 0}
.scroll{overflow-x:auto}
.chart{display:block;width:100%%;max-width:640px;height:auto;margin:0 auto}
code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);
  padding:1px 5px;border-radius:4px;border:1px solid var(--line)}
.foot{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--ink3);font-size:13px;line-height:1.7}
#cloudwrap{position:relative}
#cloud{width:100%%;height:440px;display:block;cursor:grab;touch-action:none;border-radius:8px}
#cloud.drag{cursor:grabbing}
#tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);
  padding:5px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .12s;
  white-space:nowrap;font-variant-numeric:tabular-nums}
.ctl{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.ctl button{font:inherit;font-size:13px;padding:6px 13px;border-radius:999px;cursor:pointer;
  border:1px solid var(--line);background:var(--bg);color:var(--ink2)}
.ctl button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
"""



def bars_coverage(cov, axis, w=560, rowh=30):
    """Grouped horizontal bars: coverage of an external metadata axis, by budget."""
    b, tot = cov[axis]["budgets"], cov[axis]["total"]
    r, f = cov[axis]["by"]["random"], cov[axis]["by"]["diverse"]
    h = rowh * len(b) + 26
    barw = w - 150
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
         f'aria-label="{axis} covered, random versus diverse, by budget">']
    for i, k in enumerate(b):
        yy = 18 + i * rowh
        p.append(f'<text x="0" y="{yy+11}" fill="var(--ink3)" font-size="12">k={k}</text>')
        for j, (src, col) in enumerate(((r, "var(--s1)"), (f, "var(--s2)"))):
            bw = max(1.5, src["mean"][i] / tot * barw)
            p.append(f'<rect x="46" y="{yy+j*10:.0f}" width="{bw:.1f}" height="8" rx="2.5" '
                     f'fill="{col}"/>')
            p.append(f'<text x="{46+bw+7:.1f}" y="{yy+j*10+8:.0f}" fill="var(--ink2)" '
                     f'font-size="11">{src["mean"][i]:.1f}</text>')
    p.append(f'<text x="46" y="10" fill="var(--ink3)" font-size="11">'
             f'of {tot} {axis} &#183; '
             f'<tspan fill="var(--s1)">random</tspan> vs '
             f'<tspan fill="var(--s2)">diverse</tspan></text>')
    p.append('</svg>')
    return "".join(p)


def sweep_chart(sweep, real_nn, w=520, h=250):
    """Diverging bars around zero: does injecting near-duplicates move the score?"""
    n = len(sweep)
    bw = (w - 70) / n - 12
    dmax = max(abs(r["delta"]) for r in sweep) * 1.25 or 1
    zero = h - 66
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
         f'aria-label="Change in the score when near-duplicates are injected">']
    p.append(f'<line x1="52" y1="{zero}" x2="{w-8}" y2="{zero}" stroke="var(--ink3)"/>')
    for i, r in enumerate(sweep):
        x = 62 + i * ((w - 70) / n)
        hgt = abs(r["delta"]) / dmax * (zero - 26)
        up = r["delta"] > 0
        yy = zero - hgt if up else zero
        p.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{max(hgt,1.5):.1f}" '
                 f'rx="3" fill="{"var(--neg)" if up else "var(--s3)"}"/>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{(yy-6) if up else (yy+hgt+15):.1f}" '
                 f'fill="var(--ink)" font-size="11" text-anchor="middle" '
                 f'font-weight="550">{r["delta"]:+.3f}</text>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{h-30}" fill="var(--ink3)" font-size="10" '
                 f'text-anchor="middle">{r["cos_dist"]:.4f}</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-8}" fill="var(--ink3)" font-size="11" '
             f'text-anchor="middle">cosine distance injected '
             f'(real distinct episodes differ by {real_nn:.4f})</text>')
    p.append('<text x="6" y="20" fill="var(--ink3)" font-size="11">&#916; score</text>')
    p.append('</svg>')
    return "".join(p)


def main() -> int:
    A = json.loads((RESULTS / "analysis.json").read_text())
    J = (json.loads((RESULTS / "llm_judge.json").read_text())
         if (RESULTS / "llm_judge.json").exists() else None)

    K = A["subsets"]["random"]["k"]
    R, Dv = A["subsets"]["random"], A["subsets"]["diverse"]
    Sp = A["subsets"]["spread"]
    E, dup, cov = A["economics"], A["duplication"], A["coverage"]

    sheets = {}
    for nm in ("random", "diverse"):
        f = RESULTS / "sheets" / f"{nm}.jpg.b64"
        sheets[nm] = f.read_text() if f.exists() else ""

    def subset_card(name, s, colour, sheet):
        return f"""
        <div class="card">
          <div class="herol"><span class="dot" style="background:{colour}"></span>{name}</div>
          <div class="hero">{s['vendi']:.2f}</div>
          <div style="color:var(--ink3);font-size:12px;margin-top:5px">
            effectively {s['vendi']:.1f} distinct episodes out of {s['k']}</div>
          {f'<img class="sheet" alt="One frame from each episode in the {name} subset" '
           f'src="data:image/jpeg;base64,{sheet}">' if sheet else ''}
        </div>"""

    # ---- LLM comparison
    llm = ""
    if J and "mean" in J.get("random", {}):
        d0 = J.get("diverse_temp0", {})
        per_call = J["cost_per_subset_usd"]
        sweep_calls = E["subset_scorings_performed"] * J["repeats"]
        llm = f"""
  <h2>The incumbent, measured on the same two subsets</h2>
  <div class="card">
    <div class="scroll"><table>
      <tr><th></th><th>Vendi over DINOv2</th><th>{J['model']} as a judge</th></tr>
      <tr><td>same answer twice on the same input</td>
          <td class="win">yes — identical, bit for bit</td>
          <td class="lose">no — {d0.get('spread', 0):.0f}-point spread over
              {J['repeats']} calls at temperature&nbsp;0</td></tr>
      <tr><td>survives the duplication test</td>
          <td class="win">yes — {dup['exact_clones']['delta']:+.3f} on exact clones</td>
          <td class="lose">no — {J['diverse']['mean']:.0f} &#8594;
              {J['diverse_30pct_duplicated']['mean']:.0f} when 30% of the clips are
              replaced by copies</td></tr>
      <tr><td>cost to score one subset</td>
          <td class="win">$0.00 after the index</td>
          <td>${per_call:.4f} &#183; {J['random']['latency_s_mean']:.1f}s</td></tr>
      <tr><td>auditable</td>
          <td class="win">a number you can recompute</td>
          <td>a sentence you cannot</td></tr>
    </table></div>
    <p class="note">The judge saw exactly the contact sheets above and was asked for a
      0&#8211;100 diversity rating, {J['repeats']} times each, then {J['repeats']} more at
      temperature&nbsp;0. To be fair to it: it is doing a harder job — reading semantics
      with no reference set — and it writes an explanation we cannot.
      The narrow claim is that <strong>as a ranking instrument it is not reproducible</strong>,
      and it cannot be falsified the way a metric can.</p>
  </div>

  <h2>Why this compounds</h2>
  <div class="grid3">
    <div class="card"><div class="herol">index, built once</div>
      <div class="hero sm">${E['index_cost_usd']:.3f}</div>
      <p class="note">{E['index_gpu_seconds']:.0f}s on one L4 for all
        {A['n_episodes']} episodes, {A['ms_per_frame']:.1f} ms/frame.</p></div>
    <div class="card"><div class="herol">subsets scored since</div>
      <div class="hero sm">{E['subset_scorings_performed']:,}</div>
      <p class="note">every one of them free — numpy over cached vectors, the whole
        sweep replays in under a minute on a laptop.</p></div>
    <div class="card"><div class="herol">same sweep, LLM judge</div>
      <div class="hero sm">${sweep_calls * per_call:,.0f}</div>
      <p class="note">{sweep_calls:,} calls at {J['random']['latency_s_mean']:.1f}s
        &#8594; ~{sweep_calls * J['random']['latency_s_mean'] / 3600:.0f} hours,
        and still not reproducible.</p></div>
  </div>
  <p class="note">This is the structural difference, not a tuning win: an embedding is a
    <strong>one-off index</strong>, a judge is a <strong>per-query cost</strong>. Curation is
    not one question — it is thousands of subset comparisons in a loop.</p>"""

    SLIDE_CSS = SP.CSS
    SLIDE_BODY = SP.body(A, J) if J else ""
    SLIDE_JS = SP.JS % {"data": SP.payload(A)}

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diversity Index</title>
<style>{CSS % C}
#hero{{background:#0a0a09;height:100vh;min-height:620px;display:grid;place-items:center;
  overflow:hidden;position:relative}}
#stage{{transform-origin:center center}}
#cue{{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);color:#8f8e86;
  font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;font-weight:600}}
{SLIDE_CSS}
</style></head><body>
<div id="hero"><div id="stage">{SLIDE_BODY}</div>
  <div id="cue">scroll for the evidence &#8595;</div></div>
<div class="wrap">
  <h1>The evidence</h1>
  <p class="sub">Every number on the slide above, with the measurement behind it.</p>
  <p class="meta">{A['n_episodes']} <code>cup_on_saucer</code> episodes &#183;
     DINOv2-base on a Modal L4 &#183; <code>python run_all.py</code> regenerates all of it
     in seconds, on a laptop, with no GPU and no credentials</p>

  <h2>The deliverable: a score that ranks two subsets</h2>
  <div class="grid2">
    {subset_card('Random ' + str(K), R, 'var(--s1)', sheets['random'])}
    {subset_card('Diverse ' + str(K), Dv, 'var(--s2)', sheets['diverse'])}
  </div>
  <p class="note">Same budget, same pool, same encoder — the diverse subset scores
     <strong>{Dv['vendi']/R['vendi']:.1f}&#215; higher</strong>. The thumbnails are the
     score made visible: the random grid repeats near-identical clips, the diverse grid
     does not. Selection is <strong>cluster cover</strong> (k-means, then the real episode
     nearest each centroid); the score is the Vendi Score, read as "effectively N distinct
     episodes". Cluster cover is chosen over pure farthest-point deliberately: FPS scores
     far higher ({Sp['vendi']:.2f}) but collects outliers and represents only
     {Sp['covered_pct']:.0f}% of the corpus against cluster cover's
     {Dv['covered_pct']:.0f}%. The highest score is not the best subset.</p>

  <h2>The score tracks things it was never shown</h2>
  <div class="grid2">
    <div class="card">
      <div class="herol">recording sessions covered</div>
      {bars_coverage(cov, 'recording days')}
      <p class="note">At k=16 the diverse subset reaches
        {cov['recording days']['by']['diverse']['mean'][2]:.1f} of
        {cov['recording days']['total']} recording days against random's
        {cov['recording days']['by']['random']['mean'][2]:.1f} — <strong>+{
        (cov['recording days']['by']['diverse']['mean'][2] /
         cov['recording days']['by']['random']['mean'][2] - 1) * 100:.0f}%</strong>.</p>
    </div>
    <div class="card">
      <div class="herol">prop combinations covered</div>
      {bars_coverage(cov, 'prop combos')}
      <p class="note">{cov['prop combos']['total']} distinct cup/saucer combinations exist
        in the registry metadata.</p>
    </div>
  </div>
  <p class="note">Neither the recording date nor the <code>objects</code> field is visible
    to the encoder — it only ever saw pixels. So a subset the score calls more diverse
    covering more sessions and more prop combinations is <strong>external
    evidence</strong>, not the score admiring its own geometry.</p>

  {llm}

  <h2>How it could be fooled</h2>
  <div class="card">
    {sweep_chart(dup['sweep'], dup['real_nn_cosine_dist'])}
    <p class="note"><strong>Exact duplicates cannot raise the score</strong>
      ({dup['exact_clones']['delta']:+.3f}) — that is a property of the metric, and it
      holds here. Near-duplicates are the honest caveat: jitter a copy by a cosine
      distance of 0.0094 — below the {dup['real_nn_cosine_dist']:.4f} that separates
      genuinely distinct episodes — and the score begins to inflate, reaching
      {max(r['delta'] for r in dup['sweep']):+.3f} at 0.125. We measured the threshold
      rather than claim there wasn't one. A second known limit: farthest-point is the
      wrong tool for the <em>last</em> rare category — it needs ~101 episodes to cover all
      {cov['prop combos']['total']} prop combos where random needs ~60.</p>
  </div>

  <div class="foot">
    <strong>Method.</strong> 32 uniformly spaced frames per episode &#8594; DINOv2-base
    (768-d, self-supervised, <em>no text tower</em>) on a Modal L4 &#8594; mean-pooled to one
    vector per episode &#8594; Vendi Score, the exponential of the Shannon entropy of the
    eigenvalues of the normalised cosine kernel.<br>
    Uniform sampling is a measured choice, not a default: four CPU keyframe-selection
    policies were tried and all lost to <code>np.linspace</code>, and keyframe pooling lost
    to mean pooling (+0.319 vs +0.352 separation). Both experiments are in the repo.
  </div>
</div>
<script>
(function(){{
  var st=document.getElementById('stage');
  function fit(){{
    var h=document.getElementById('hero').clientHeight;
    st.style.transform='scale('+Math.min(innerWidth/1600,h/900)+')';
  }}
  addEventListener('resize',fit); fit();
}})();
{SLIDE_JS}</script>
</body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  random VS {R['vendi']}  diverse VS {Dv['vendi']}  "
          f"index ${E['index_cost_usd']:.4f}  scorings {E['subset_scorings_performed']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
