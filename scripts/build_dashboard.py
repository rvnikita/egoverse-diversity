"""Render the Track 2 dashboard: two subsets, scored, compared, and stress-tested.

Self-contained HTML — inline SVG, inline base64 thumbnails, no external requests, no
JS libraries. Reads only the committed artefacts in results/.

    .venv/bin/python scripts/build_dashboard.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "dashboard.html"

# Validated 3-slot categorical palette (all-pairs, light and dark) — see dataviz skill.
C = {
    "pool_l": "#a8a7a1", "pool_d": "#5c5b56",
    "s1_l": "#2a78d6", "s1_d": "#3987e5",     # random
    "s2_l": "#eb6834", "s2_d": "#d95926",     # diverse
    "s3_l": "#1baf7a", "s3_d": "#199e70",     # third series where needed
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
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 72px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--ink2);margin:0 0 4px;font-size:16px}
.meta{color:var(--ink3);font-size:13px;margin:14px 0 0}
.thesis{margin:26px 0 0;padding:18px 20px;background:var(--surface);
  border:1px solid var(--line);border-left:3px solid var(--s2);border-radius:8px;
  font-size:17px;line-height:1.5}
h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);
  margin:44px 0 14px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.hero{font-size:44px;font-weight:650;letter-spacing:-.03em;line-height:1}
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
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
td:first-child{font-variant-numeric:normal}
.win{color:var(--s3);font-weight:650}
.lose{color:var(--neg);font-weight:650}
.note{color:var(--ink2);font-size:13.5px;line-height:1.6;margin:12px 0 0}
.scroll{overflow-x:auto}
.chart{display:block;width:100%%;max-width:640px;height:auto;margin:0 auto}
code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);
  padding:1px 5px;border-radius:4px;border:1px solid var(--line)}
.foot{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--ink3);font-size:13px}
"""


# --------------------------------------------------------------------- svg helpers


def scatter_pca(proj, idx_rand, idx_div, w=620, h=400):
    x, y = np.array(proj["x"]), np.array(proj["y"])
    pad = 34
    # Use the real data range. Clamping would pile outliers into a false column on the
    # frame edge — and outliers are the entire subject of this chart.
    xlo, xhi = x.min() - .15, x.max() + .15
    ylo, yhi = y.min() - .15, y.max() + .15
    sx = lambda v: pad + (v - xlo) / (xhi - xlo) * (w - pad - 16)  # noqa: E731
    sy = lambda v: h - pad - (v - ylo) / (yhi - ylo) * (h - pad - 18)  # noqa: E731

    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
         f'aria-label="PCA projection of all episodes with the two subsets overlaid">']
    p.append(f'<rect x="{pad}" y="14" width="{w-pad-14}" height="{h-pad-14}" '
             f'fill="none" stroke="var(--grid)"/>')
    sel = set(idx_rand) | set(idx_div)
    for i in range(len(x)):
        if i not in sel:
            p.append(f'<circle cx="{sx(x[i]):.1f}" cy="{sy(y[i]):.1f}" r="2.1" '
                     f'fill="var(--pool)" opacity=".38"/>')
    for i in idx_rand:
        p.append(f'<circle cx="{sx(x[i]):.1f}" cy="{sy(y[i]):.1f}" r="5" fill="var(--s1)" '
                 f'stroke="var(--surface)" stroke-width="2"/>')
    for i in idx_div:
        p.append(f'<circle cx="{sx(x[i]):.1f}" cy="{sy(y[i]):.1f}" r="5" fill="var(--s2)" '
                 f'stroke="var(--surface)" stroke-width="2"/>')
    p.append(f'<text x="{pad}" y="{h-10}" fill="var(--ink3)" font-size="11">PC1</text>')
    p.append(f'<text x="6" y="24" fill="var(--ink3)" font-size="11">PC2</text>')
    p.append('</svg>')
    return "".join(p)


# Colour follows the ENTITY, never its rank: `fps` is orange on every chart because it is
# the same selector as the orange "diverse" subset above. Colouring by rank would paint fps
# green on the diversity chart and red on the AUROC chart — the same thing, two identities.
BAR_COLOUR = {"fps": "var(--s2)", "random": "var(--s1)",
              "kmedoid": "var(--s3)", "matched_random": "var(--pool)"}


def hbars(rows, value_key, label_key, fmt="{:.3f}", w=520, rowh=34, hi=None, best="max"):
    """Horizontal bars, direct-labelled. One measure only — never two scales."""
    vals = [r[value_key] for r in rows]
    vmax = max(vals) * 1.18
    h = rowh * len(rows) + 12
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for i, r in enumerate(rows):
        yy = i * rowh + 8
        bw = max(2.0, r[value_key] / vmax * (w - 210))
        col = BAR_COLOUR.get(r[label_key], "var(--s1)")
        p.append(f'<text x="0" y="{yy+15}" fill="var(--ink2)" font-size="13">'
                 f'{r[label_key]}</text>')
        p.append(f'<rect x="128" y="{yy+3}" width="{bw:.1f}" height="17" rx="4" fill="{col}"/>')
        p.append(f'<text x="{128+bw+9:.1f}" y="{yy+16}" fill="var(--ink)" font-size="13" '
                 f'font-weight="550">{fmt.format(r[value_key])}</text>')
    p.append('</svg>')
    return "".join(p)


def scatter_corr(df, w=540, h=360):
    """Vendi vs AUROC, one series. Cluster centroids direct-labelled, trend line drawn."""
    xs, ys, sels = df.vendi.to_numpy(), df.auroc.to_numpy(), df.selector.to_numpy()
    pad_l, pad_b = 46, 34
    xlo, xhi = xs.min() * .96, xs.max() * 1.04
    ylo, yhi = min(ys.min() * .97, .45), max(ys.max() * 1.03, .85)
    sx = lambda v: pad_l + (v - xlo) / (xhi - xlo) * (w - pad_l - 16)  # noqa: E731
    sy = lambda v: h - pad_b - (v - ylo) / (yhi - ylo) * (h - pad_b - 16)  # noqa: E731

    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
         f'aria-label="Vendi score against downstream AUROC across all runs">']
    for gy in np.linspace(ylo, yhi, 5):
        p.append(f'<line x1="{pad_l}" y1="{sy(gy):.1f}" x2="{w-16}" y2="{sy(gy):.1f}" '
                 f'stroke="var(--grid)"/>')
        p.append(f'<text x="{pad_l-8}" y="{sy(gy)+4:.1f}" fill="var(--ink3)" font-size="11" '
                 f'text-anchor="end">{gy:.2f}</text>')
    for xv, yv in zip(xs, ys):
        p.append(f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="3.6" fill="var(--s1)" '
                 f'opacity=".5"/>')
    b, a = np.polyfit(xs, ys, 1)
    p.append(f'<line x1="{sx(xlo):.1f}" y1="{sy(a+b*xlo):.1f}" x2="{sx(xhi):.1f}" '
             f'y2="{sy(a+b*xhi):.1f}" stroke="var(--neg)" stroke-width="2" '
             f'stroke-dasharray="6 4"/>')
    for s in sorted(set(sels.tolist())):
        m = sels == s
        cx, cy = sx(xs[m].mean()), sy(ys[m].mean())
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="var(--s2)" '
                 f'stroke="var(--surface)" stroke-width="2"/>')
        anchor = "end" if cx > w * .55 else "start"
        dx = -10 if anchor == "end" else 10
        p.append(f'<text x="{cx+dx:.1f}" y="{cy+4:.1f}" fill="var(--ink)" font-size="12" '
                 f'font-weight="600" text-anchor="{anchor}">{s}</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-6}" fill="var(--ink3)" font-size="11" '
             f'text-anchor="middle">Vendi score of the selected subset →</text>')
    # rotated, so it cannot collide with the topmost y tick label
    p.append(f'<text transform="translate(13,{h/2:.0f}) rotate(-90)" fill="var(--ink3)" '
             f'font-size="11" text-anchor="middle">AUROC of a model trained on it →</text>')
    p.append('</svg>')
    return "".join(p)


def sweep_chart(sweep, real_nn, w=520, h=250):
    """Diverging bars around zero: does injecting near-duplicates raise the score?"""
    rows = [r for r in sweep]
    n = len(rows)
    bw = (w - 70) / n - 12
    dmax = max(abs(r["delta"]) for r in rows) * 1.25 or 1
    zero = h - 66
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
         f'aria-label="Change in Vendi score when near-duplicates are injected">']
    p.append(f'<line x1="52" y1="{zero}" x2="{w-8}" y2="{zero}" stroke="var(--ink3)"/>')
    for i, r in enumerate(rows):
        x = 62 + i * ((w - 70) / n)
        hgt = abs(r["delta"]) / dmax * (zero - 26)
        up = r["delta"] > 0
        yy = zero - hgt if up else zero
        col = "var(--neg)" if up else "var(--s3)"
        p.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{max(hgt,1.5):.1f}" '
                 f'rx="3" fill="{col}"/>')
        lab_y = yy - 6 if up else yy + hgt + 15
        p.append(f'<text x="{x+bw/2:.1f}" y="{lab_y:.1f}" fill="var(--ink)" font-size="11" '
                 f'text-anchor="middle" font-weight="550">{r["delta"]:+.3f}</text>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{h-30}" fill="var(--ink3)" font-size="10" '
                 f'text-anchor="middle">{r["cos_dist"]:.4f}</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-8}" fill="var(--ink3)" font-size="11" '
             f'text-anchor="middle">cosine distance injected  '
             f'(real distinct episodes differ by {real_nn:.4f})</text>')
    p.append(f'<text x="6" y="20" fill="var(--ink3)" font-size="11">Δ Vendi</text>')
    p.append('</svg>')
    return "".join(p)


# --------------------------------------------------------------------------- page


def main() -> int:
    A = json.loads((RESULTS / "analysis.json").read_text())
    strat = pd.read_csv(RESULTS / "selection_stratified.csv")
    sess = (pd.read_csv(RESULTS / "selection_session.csv")
            if (RESULTS / "selection_session.csv").exists() else None)

    K = A["subsets"]["random"]["k"]
    m32 = strat[(strat.k == K) & (strat.frames == 32)]
    agg = (m32.groupby("selector")
           .agg(vendi=("vendi", "mean"), fails=("n_fail", "mean"),
                combo=("combo_cov", "mean"), auroc=("auroc", "mean"),
                bal=("bal_acc", "mean")).reset_index())
    order = ["random", "kmedoid", "matched_random", "fps"]
    agg["o"] = agg.selector.map({s: i for i, s in enumerate(order)})
    agg = agg.sort_values("o")
    rows = agg.to_dict("records")

    from scipy.stats import spearmanr
    rho, pv = spearmanr(m32.vendi, m32.auroc)

    R, Dv = A["subsets"]["random"], A["subsets"]["diverse"]
    sheets = {}
    for nm in ("random", "diverse"):
        f = RESULTS / "sheets" / f"{nm}.jpg.b64"
        sheets[nm] = f.read_text() if f.exists() else ""

    knn = A["label_free_ranking"]["knn20_dist"]
    fa = A["failures_atypical"]
    dup = A["duplication"]

    def card_subset(name, s, colour, sheet):
        return f"""
        <div class="card">
          <div class="herol"><span class="dot" style="background:{colour}"></span>{name}</div>
          <div class="hero">{s['vendi']:.2f}</div>
          <div style="color:var(--ink3);font-size:12px;margin-top:4px">Vendi score
            — effectively {s['vendi']:.1f} distinct episodes out of {s['k']}</div>
          <div style="margin-top:16px">
            <div class="kv"><span>failure demos included</span>
              <span>{s['failures']} / {s['k']} &nbsp;({s['failure_pct']:.0f}%)</span></div>
            <div class="kv"><span>prop combos covered</span>
              <span>{s['combos']} / {A['n_combos']}</span></div>
          </div>
          {f'<img class="sheet" alt="Final frame of each episode in the {name} subset" '
           f'src="data:image/jpeg;base64,{sheet}">' if sheet else ''}
        </div>"""

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atypicality Dashboard</title>
<style>{CSS % C}</style></head><body>
<div class="wrap">
  <h1>Diversity, measured without text</h1>
  <p class="sub">Track 2 · EgoVerse Data Optimization &amp; Evaluation Suite</p>
  <p class="meta">{A['n_episodes']} <code>cup_on_saucer</code> episodes ·
     DINOv2-base on a Modal L4, no text encoder anywhere ·
     {A['ms_per_frame']:.1f} ms/frame, {A['gpu_seconds_total']:.0f} s of GPU total ·
     every number below regenerates on a laptop in under a minute</p>

  <div class="thesis">
    A diversity score ranks subsets. It does <strong>not</strong> tell you which subset to
    train on — maximise it blindly and you get the worst one. One axis,
    <strong>atypicality</strong>, drives all three: it inflates the diversity score,
    it concentrates failure demos <strong>{knn[f'top{K}_enrichment']}×</strong>,
    and it makes the subset <em>worse</em> to train on
    (ρ&nbsp;=&nbsp;{rho:+.2f}, p&nbsp;&lt;&nbsp;0.001).
  </div>

  <h2>The score ranks two subsets</h2>
  <div class="grid2">
    {card_subset('Random ' + str(K), R, 'var(--s1)', sheets['random'])}
    {card_subset('Diverse ' + str(K) + ' (farthest-point)', Dv, 'var(--s2)', sheets['diverse'])}
  </div>
  <p class="note">Same budget, same pool, same encoder. The diverse subset scores
     <strong>{Dv['vendi']/R['vendi']:.1f}× higher</strong> and its thumbnails are visibly
     less redundant — the random grid repeats near-identical clips.</p>

  <h2>Where they sit in embedding space</h2>
  <div class="card">
    {scatter_pca(A['projection'], R['idx'], Dv['idx'])}
    <p class="note"><span class="dot" style="background:var(--pool)"></span>all
      {A['n_episodes']} episodes &nbsp;
      <span class="dot" style="background:var(--s1)"></span>random {K} &nbsp;
      <span class="dot" style="background:var(--s2)"></span>diverse {K}
      &nbsp;— 2D PCA. Random clumps in the mode; farthest-point spreads to the rim.</p>
  </div>

  <h2>Is the score meaningful, or just measuring itself?</h2>
  <div class="grid2">
    <div class="card">
      <div class="herol">Diversity score</div>
      {hbars(rows, 'vendi', 'selector')}
      <p class="note">Farthest-point wins by construction — it maximises the
        same spread the score measures.</p>
    </div>
    <div class="card">
      <div class="herol">Downstream AUROC of a model trained on that subset</div>
      {hbars(rows, 'auroc', 'selector')}
      <p class="note">…and loses badly. <code>matched_random</code> holds failure
        count identical to <code>fps</code>, so this is <strong>not</strong> class balance.</p>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    {scatter_corr(m32)}
    <p class="note">Every run at k={K}. Higher diversity, lower downstream utility:
      Spearman ρ = <strong>{rho:+.3f}</strong> (p = {pv:.1e}), measured
      <em>within</em> a single budget so subset size cannot drive the correlation.
      {"Holds on a session-split too (ρ = %+.2f), where whole recording days are held out."
       % spearmanr(sess[(sess.k==K)&(sess.frames==32)].vendi,
                   sess[(sess.k==K)&(sess.frames==32)].auroc)[0] if sess is not None else ""}</p>
  </div>

  <h2>What the score buys you, label-free</h2>
  <div class="card">
    <div class="scroll"><table>
      <tr><th>label-free score</th><th>AUROC vs failure</th>
          <th>failures in top {K}</th><th>enrichment</th></tr>
      {"".join(f"<tr><td><code>{k}</code></td><td>{v['auroc']:.3f}</td>"
               f"<td>{v[f'top{K}_failures']} / {K}</td>"
               f"<td class='{'win' if v[f'top{K}_enrichment']>2 else ''}'>"
               f"{v[f'top{K}_enrichment']}×</td></tr>"
               for k, v in A['label_free_ranking'].items())}
    </table></div>
    <p class="note">The registry has <strong>no usable outcome labels</strong>
      (<code>eval_success</code> is a dataclass default on all 439,053 rows); these came from
      task-name suffixes on one lab's episodes. Nothing above uses them —
      the scores are pure geometry, and they still rank failures.
      <strong>Note the trap:</strong> <code>dist_to_centroid</code> and
      <code>knn20_dist</code> have near-identical AUROC but their top-{K} differ
      7 vs {knn[f'top{K}_failures']}. AUROC alone would pick the wrong one.</p>
    <p class="note">Mechanism: failures genuinely sit further out —
      Vendi {fa['vendi_failures']} vs {fa['vendi_successes_matched_n']}±{fa['vendi_successes_sd']}
      for the same number of successes, mean distance to the mode
      {fa['centroid_dist_failures']:.4f} vs {fa['centroid_dist_successes']:.4f},
      Mann-Whitney p = {fa['mannwhitney_p']:.1e}.</p>
  </div>

  <h2>What it fails at</h2>
  <div class="card">
    {sweep_chart(dup['sweep'], dup['real_nn_cosine_dist'])}
    <p class="note"><strong>Exact duplicates cannot raise the score</strong>
      (Δ {dup['exact_clones']['delta']:+.3f}) — that is provable and it holds here.
      But near-duplicates <em>can</em>. Perturb a clone by a cosine distance of
      0.0094 — well under the {dup['real_nn_cosine_dist']:.4f} that separates genuinely
      distinct episodes — and the score starts to inflate, reaching
      {max(r['delta'] for r in dup['sweep']):+.3f} at 0.125.
      <strong>The score is gameable by anyone who submits jittered copies.</strong>
      We measured the threshold rather than claiming there isn't one.</p>
  </div>

  <div class="foot">
    Method: 32 uniform frames per episode → DINOv2-base (768-d, self-supervised, no text
    tower) on a Modal L4 → mean-pooled per episode → Vendi Score (exponential of the
    Shannon entropy of the eigenvalues of the normalised cosine kernel).
    Audit classifier is PCA-16 + L2 logistic regression on a <em>different</em>
    representation (first 4 + last 4 frames), frozen a priori and identical across
    selectors. 20 seeds per selector per budget.
  </div>
</div></body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  rho={rho:+.3f} p={pv:.2e}  |  random VS {R['vendi']} vs diverse {Dv['vendi']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
