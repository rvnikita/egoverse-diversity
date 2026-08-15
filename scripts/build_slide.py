"""The one summary slide. 1600x900, self-contained, screenshot-ready.

    .venv/bin/python scripts/build_slide.py
"""

from __future__ import annotations

import io
import json
import pathlib

import segno

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "slide.html"
REPO = "https://github.com/rvnikita/egoverse-diversity"


def qr_svg(url: str) -> str:
    buf = io.BytesIO()  # segno writes bytes even for SVG
    segno.make(url, error="m").save(buf, kind="svg", scale=1, border=1,
                                    dark="#0b0b0b", light=None, xmldecl=False, svgns=True)
    return buf.getvalue().decode()


def main() -> int:
    A = json.loads((RESULTS / "analysis.json").read_text())
    J = json.loads((RESULTS / "llm_judge.json").read_text())
    R, Dv, E = A["subsets"]["random"], A["subsets"]["diverse"], A["economics"]
    cov = A["coverage"]
    per_call = J["cost_per_subset_usd"]
    sweep_calls = E["subset_scorings_performed"] * J["repeats"]
    d0 = J["diverse_temp0"]

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Diversity Index — slide</title>
<style>
*,*::before,*::after{{box-sizing:border-box}}
html,body{{margin:0;background:#e8e8e4;overflow:hidden}}
/* The slide is authored at a fixed 1600x900 and scaled to whatever it is opened on, so
   it is identical in a browser, in a screenshot, and on a projector. */
#stage{{position:absolute;top:50%;left:50%;transform-origin:center center}}
.slide{{width:1600px;height:900px;background:#fcfcfb;color:#0b0b0b;padding:52px 60px;
  font:16px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column;position:relative}}
h1{{font-size:52px;line-height:1.03;margin:0;letter-spacing:-.03em;font-weight:680}}
.tag{{font-size:15px;color:#7c7b76;margin:12px 0 0;letter-spacing:.01em}}
.lede{{font-size:20px;color:#52514e;margin:16px 0 0;max-width:58ch;line-height:1.45}}
.cols{{display:grid;grid-template-columns:1.02fr 1fr;gap:42px;margin-top:30px;flex:1}}
h2{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#7c7b76;
  margin:0 0 12px;font-weight:650}}
.pipe{{font:13px/1.85 ui-monospace,SFMono-Regular,Menlo,monospace;color:#52514e;
  background:#f4f3f0;border:1px solid #e3e2dd;border-radius:8px;padding:14px 16px;
  white-space:pre}}
table{{width:100%;border-collapse:collapse;font-size:14.5px}}
td{{padding:9px 8px 9px 0;border-bottom:1px solid #e3e2dd;vertical-align:top}}
td:first-child{{color:#52514e;width:38%}}
.ok{{color:#0d7a52;font-weight:650}}
.no{{color:#c62f2e;font-weight:650}}
.tiles{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}}
.tile{{background:#f4f3f0;border:1px solid #e3e2dd;border-radius:9px;padding:14px 16px}}
.tv{{font-size:31px;font-weight:670;letter-spacing:-.025em;line-height:1}}
.tl{{font-size:11px;color:#7c7b76;text-transform:uppercase;letter-spacing:.06em;
  margin-bottom:7px;font-weight:600}}
.tn{{font-size:12px;color:#52514e;margin-top:7px;line-height:1.45}}
.scores{{display:flex;gap:36px;align-items:flex-end;margin-top:4px}}
.sc b{{font-size:44px;font-weight:680;letter-spacing:-.03em;display:block;line-height:1}}
.sc span{{font-size:12.5px;color:#7c7b76}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}}
.foot{{display:flex;align-items:center;gap:18px;margin-top:22px;padding-top:18px;
  border-top:1px solid #e3e2dd}}
.foot svg{{width:96px;height:96px;display:block}}
.repo{{font:15px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}}
.real{{font-size:12.5px;color:#52514e;margin-top:5px;line-height:1.5}}
</style></head><body>
<div id="stage"><div class="slide">
  <div>
    <h1>Score diversity once. Query it forever.</h1>
    <p class="tag"><strong>Track 2 — Quantitative Diversity Measurement</strong>
       &nbsp;·&nbsp; team rvnikita &nbsp;·&nbsp; EgoVerse, NYC, 15 Aug 2026</p>
    <p class="lede">A diversity score for video subsets with <strong>no text encoder
       anywhere</strong>. It ranks two subsets, returns the same answer every time, and
       once the index exists every further question is free.</p>
  </div>

  <div class="cols">
    <div>
      <h2>Pipeline</h2>
      <div class="pipe">32 uniform frames  ─▶  DINOv2-base (768-d, no text tower)
                          on a Modal L4 · 8.8 ms/frame
                                   │
                    mean-pool ─▶ one vector per episode
                                   │
        ┌──────────────────────────┴──────────────────────────┐
   Vendi Score                                    farthest-point
   "effectively N distinct"                       subset selection
        └──────────────────────────┬──────────────────────────┘
                          score · rank · dashboard</div>

      <h2 style="margin-top:26px">The deliverable — two subsets, ranked</h2>
      <div class="scores">
        <div class="sc"><b><span class="dot" style="background:#2a78d6"></span>{R['vendi']:.2f}</b>
          <span>random {R['k']}</span></div>
        <div class="sc"><b><span class="dot" style="background:#eb6834"></span>{Dv['vendi']:.2f}</b>
          <span>diverse {Dv['k']} &nbsp;·&nbsp; {Dv['vendi']/R['vendi']:.1f}× higher</span></div>
      </div>
      <p class="real" style="margin-top:14px">Validated against metadata the encoder never
        saw: at k=16 the diverse subset covers
        <strong>{cov['recording days']['by']['diverse']['mean'][2]:.1f} of
        {cov['recording days']['total']}</strong> recording sessions vs random's
        {cov['recording days']['by']['random']['mean'][2]:.1f}.</p>
    </div>

    <div>
      <h2>vs. the LLM-as-a-judge it replaces &nbsp;<span style="text-transform:none;
        letter-spacing:0;font-weight:500">(gpt-4o, same images, measured)</span></h2>
      <table>
        <tr><td>Same answer twice, same input</td>
            <td><span class="ok">identical</span> — bit for bit</td>
            <td><span class="no">{d0['spread']:.0f}-pt spread</span> at temperature 0</td></tr>
        <tr><td>Survives the duplication test</td>
            <td><span class="ok">yes</span> — {A['duplication']['exact_clones']['delta']:+.3f}</td>
            <td><span class="no">no</span> — {J['diverse']['mean']:.0f}→{J['diverse_30pct_duplicated']['mean']:.0f}
                with 30% copies</td></tr>
        <tr><td>Cost per subset scored</td>
            <td><span class="ok">$0.00</span> after the index</td>
            <td>${per_call:.4f} · {J['random']['latency_s_mean']:.1f}s</td></tr>
      </table>

      <h2 style="margin-top:26px">Why it compounds</h2>
      <div class="tiles">
        <div class="tile"><div class="tl">index, once</div>
          <div class="tv">${E['index_cost_usd']:.3f}</div>
          <div class="tn">{E['index_gpu_seconds']:.0f} s on one L4,
            all {A['n_episodes']} episodes</div></div>
        <div class="tile"><div class="tl">subsets scored</div>
          <div class="tv">{E['subset_scorings_performed']:,}</div>
          <div class="tn">every one free — numpy on cached vectors</div></div>
        <div class="tile"><div class="tl">same sweep, LLM</div>
          <div class="tv">~${sweep_calls*per_call:,.0f}</div>
          <div class="tn">{sweep_calls:,} calls → ~3 h, still not reproducible</div></div>
      </div>
      <p class="real"><strong>An embedding is a one-off index; a judge is a per-query
        cost.</strong> Curation is never one question.</p>
    </div>
  </div>

  <div class="foot">
    {qr_svg(REPO)}
    <div>
      <div class="repo">{REPO.replace('https://', '')}</div>
      <div class="real"><strong>All real, nothing mocked:</strong>
        {A['n_episodes']} real EgoVerse episodes · real DINOv2 on a real Modal L4 ·
        real gpt-4o calls. <code>python run_all.py</code> reproduces every number on this
        slide in 6 s on a laptop — no GPU, no credentials, no network.</div>
      <div class="real"><strong>Known limits:</strong> near-duplicates jittered past a
        cosine distance of 0.0094 inflate the score (measured); farthest-point is the wrong
        tool for the last rare category; one task, one lab, one scene.</div>
    </div>
  </div>
</div></div>
<script>
(function(){{
  var st=document.getElementById('stage');
  function fit(){{
    var k=Math.min(innerWidth/1600, innerHeight/900);
    st.style.transform='translate(-50%,-50%) scale('+k+')';
  }}
  addEventListener('resize',fit); fit();
}})();
</script>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
