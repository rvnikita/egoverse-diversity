"""The one summary slide: the two approaches as chains, then one shared 3D result.

Both subsets live on the SAME rotating cloud so the comparison is direct: every episode is
coloured by WHICH pick represents it (within the measured representation radius), so the
orange territory a diverse subset reaches — and random misses — is visible directly.

Every number is read from results/*.json, so the slide cannot drift from the measurements.
1600x900, scales to any viewport, fully self-contained.

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


def qr_svg(url: str, scale: int = 3) -> str:
    """segno emits no viewBox, so render at native size rather than resizing in CSS."""
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=scale, border=2,
                                    dark="#ffffff", light=None, xmldecl=False, svgns=True)
    return buf.getvalue().decode()


CSS = """
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;background:#0a0a09;overflow:hidden}
#stage{position:absolute;top:50%;left:50%;transform-origin:center center}
.slide{width:1600px;height:900px;background:#141413;color:#fff;padding:34px 50px 24px;
  font:16px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column;position:relative;overflow:hidden}
.slide::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(1000px 560px at 40% 60%,rgba(217,89,38,.06),transparent 70%)}
.top{display:flex;justify-content:space-between;align-items:flex-start;flex:none}
h1{font-size:40px;line-height:1.03;margin:0;letter-spacing:-.032em;font-weight:680}
h1 em{font-style:normal;color:#d95926}
.q{font-size:16.5px;color:#c3c2b7;margin:9px 0 0}
.q b{color:#fff;font-weight:640}
.tag{font-size:12px;color:#8f8e86;text-align:right;line-height:1.75;letter-spacing:.02em}
.tag b{color:#c3c2b7;font-weight:600}

.lanes{display:flex;flex-direction:column;gap:9px;margin-top:16px;flex:none}
.lane{display:grid;grid-template-columns:118px 1fr 268px;align-items:center;gap:16px;
  background:#1a1a19;border:1px solid #33322e;border-radius:10px;padding:11px 16px}
.lane.ours{border-color:#5c4030;background:
  linear-gradient(90deg,rgba(217,89,38,.10),rgba(217,89,38,.02) 55%,transparent),#1a1a19}
.who{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;font-weight:700;
  line-height:1.4}
.who span{display:block;font-size:10px;letter-spacing:.03em;text-transform:none;
  color:#8f8e86;font-weight:500;margin-top:2px}
.chain{display:flex;align-items:center;flex-wrap:nowrap}
.node{background:#242320;border:1px solid #3d3b36;border-radius:6px;padding:6px 10px;
  font-size:12px;color:#e8e7e0;white-space:nowrap;line-height:1.2;text-align:center}
.node b{display:block;font-size:9.5px;color:#8f8e86;font-weight:600;letter-spacing:.04em;
  text-transform:uppercase;margin-top:1px}
.node.hot{background:#d95926;border-color:#d95926;color:#141413;font-weight:650}
.node.hot b{color:#5c2810}
.node.cold{background:#2b3a4d;border-color:#3987e5;color:#dbe8f8}
.node.cold b{color:#8fb4e0}
.arw{color:#6b6a64;font-size:14px;padding:0 7px;flex:none}
.loop{color:#e66767;font-size:11.5px;font-weight:650;padding-left:10px;line-height:1.25}
.fan{display:flex;flex-direction:column;gap:2px;padding-left:9px}
.fan i{font-style:normal;font-size:10px;color:#42c08a;font-weight:650;line-height:1.15}
.cost{text-align:right;font-size:12px;line-height:1.55;color:#c3c2b7}
.bad{color:#e66767}.good{color:#42c08a}
.big{font-size:21px;font-weight:700;letter-spacing:-.02em;display:block;line-height:1.15}

.main{display:grid;grid-template-columns:1fr 400px;gap:18px;margin-top:13px;flex:1;
  min-height:0}
.viz{background:#1a1a19;border:1px solid #33322e;border-radius:11px;padding:10px 14px 6px;
  display:flex;flex-direction:column;min-height:0;position:relative}
.vizh{display:flex;justify-content:space-between;align-items:baseline;flex:none}
.vizt{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#8f8e86;
  font-weight:680}
.leg{font-size:11.5px;color:#8f8e86}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 5px 0 12px}
canvas{width:100%;flex:1;display:block;min-height:0}
.vizf{font-size:11.5px;color:#8f8e86;text-align:center;line-height:1.3;flex:none;
  padding-bottom:2px}

.panel{display:flex;flex-direction:column;gap:9px;min-height:0}
.row{background:#1a1a19;border:1px solid #33322e;border-radius:10px;padding:12px 15px}
.row.win{border-color:#5c4030}
.rh{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.rn{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;font-weight:680}
.rv{font-size:26px;font-weight:700;letter-spacing:-.03em;line-height:1}
.rv small{font-size:10.5px;color:#8f8e86;font-weight:500;letter-spacing:0;
  display:block;text-align:right;margin-top:1px}
.bar{height:11px;border-radius:6px;background:#2b2a26;overflow:hidden;margin-top:3px}
.bar i{display:block;height:100%;border-radius:6px}
.bl{display:flex;justify-content:space-between;font-size:11.5px;color:#8f8e86;
  margin-top:5px}
.bl b{color:#fff;font-weight:650}
.warn{background:#221c18;border:1px solid #4a3327;border-radius:10px;padding:10px 14px;
  font-size:11.5px;color:#c3c2b7;line-height:1.45}
.warn b{color:#e6a067;font-weight:680}
.blue{color:#3987e5}.orng{color:#d95926}

.foot{display:grid;grid-template-columns:1fr 1fr 1fr 226px;gap:18px;margin-top:12px;
  padding-top:11px;border-top:1px solid #33322e;align-items:center;flex:none}
.fl{font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;color:#8f8e86;
  font-weight:680;margin-bottom:4px}
.ft{font-size:12px;color:#c3c2b7;line-height:1.4}
.ft b{color:#fff;font-weight:650}
.qr{display:flex;gap:11px;align-items:center;justify-content:flex-end}
.qr svg{display:block;flex:none}
.repo{font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:#fff;font-weight:600}
.repos{font-size:10px;color:#8f8e86;line-height:1.4;margin-top:3px}
"""

JS = """
(function(){
  var st=document.getElementById('stage');
  function fit(){var k=Math.min(innerWidth/1600, innerHeight/900);
    st.style.transform='translate(-50%%,-50%%) scale('+k+')';}
  addEventListener('resize',fit); fit();
})();
(function(){
  var D=%(data)s;
  var cv=document.getElementById('cloud'), ctx=cv.getContext('2d');
  var A={}, B={};
  D.r.forEach(function(i){A[i]=1;}); D.d.forEach(function(i){B[i]=1;});
  var W=0,H=0,ay=0.4,ax=-0.26;
  function size(){var r=cv.getBoundingClientRect(),dpr=2;
    W=r.width;H=r.height;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);}
  function draw(){
    if(!W)return;
    ctx.clearRect(0,0,W,H);
    var ca=Math.cos(ay),sa=Math.sin(ay),cb=Math.cos(ax),sb=Math.sin(ax);
    var s=Math.min(W,H)*0.315, cx=W/2, cy=H/2, pts=[];
    for(var i=0;i<D.x.length;i++){
      var x=D.x[i],y=D.y[i],z=D.z[i];
      var x1=x*ca+z*sa, z1=-x*sa+z*ca;
      var y1=y*cb-z1*sb, z2=y*sb+z1*cb;
      var p=2.6/(2.6+z2*0.3);
      pts.push({X:cx+x1*s*p,Y:cy-y1*s*p,z:z2,k:D.k[i],pick:(A[i]||B[i]),b:B[i],p:p});
    }
    pts.sort(function(u,v){return v.z-u.z;});
    var COL=['#3a3934','#7d7c74','#3987e5','#d95926'];   // neither, both, random, diverse
    var ALP=[0.30,0.42,0.95,0.95], RAD=[1.9,2.1,3.4,3.4];
    for(var j=0;j<pts.length;j++){
      var q=pts[j];
      ctx.beginPath();
      ctx.arc(q.X,q.Y,(q.pick?5.6:RAD[q.k])*q.p,0,6.2832);
      ctx.fillStyle=q.pick?(q.b?'#d95926':'#3987e5'):COL[q.k];
      ctx.globalAlpha=q.pick?1:ALP[q.k];
      ctx.fill();
      if(q.pick){ctx.globalAlpha=1;ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();}
    }
    ctx.globalAlpha=1;
  }
  size();
  function frame(){ay+=0.0021;draw();requestAnimationFrame(frame);}
  addEventListener('resize',function(){size();draw();});
  frame();
})();
"""


def main() -> int:
    A = json.loads((RESULTS / "analysis.json").read_text())
    J = json.loads((RESULTS / "llm_judge.json").read_text())
    R, Dv, Sp = A["subsets"]["random"], A["subsets"]["diverse"], A["subsets"]["spread"]
    E, cov, pr = A["economics"], A["coverage"], A["projection"]
    per = J["cost_per_subset_usd"]
    d0 = J["diverse_temp0"]
    enr = A["label_free_ranking"]["knn20_dist"]
    n_q = E["subset_scorings_performed"]
    tau = A["coverage_radius"]["tau"]

    # Colour every episode by WHO REPRESENTS IT, computed in the true 768-d space rather
    # than approximated in the projection: 0 neither, 1 both, 2 random only, 3 diverse only.
    cr, cd = R["covered"], Dv["covered"]
    klass = [(3 if d and not r else 2 if r and not d else 1 if r and d else 0)
             for r, d in zip(cr, cd)]
    excl_r = sum(1 for k in klass if k == 2)
    excl_d = sum(1 for k in klass if k == 3)
    both = sum(1 for k in klass if k == 1)
    neither = sum(1 for k in klass if k == 0)
    data = json.dumps({"x": pr["x"], "y": pr["y"], "z": pr["z"],
                       "k": klass, "r": R["idx"], "d": Dv["idx"]},
                      separators=(",", ":"))
    arw = '<span class="arw">&#10230;</span>'

    def row(name, s, colour, cls, note):
        return f"""
      <div class="row {cls}">
        <div class="rh"><div class="rn" style="color:{colour}">{name}</div>
          <div class="rv" style="color:{colour}">{s['vendi']:.2f}
            <small>Vendi score</small></div></div>
        <div class="bar"><i style="width:{s['covered_pct']:.0f}%;background:{colour}"></i></div>
        <div class="bl"><span>represents <b>{s['covered_pct']:.0f}%</b> of the corpus</span>
          <span>{note}</span></div>
      </div>"""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Diversity Index — slide</title>
<style>{CSS}</style></head><body>
<div id="stage"><div class="slide">

  <div class="top">
    <div>
      <h1>Score diversity once. <em>Query it forever.</em></h1>
      <p class="q">439,053 episodes. You can train on a few hundred.
        <b>Which ones — and how do you prove it?</b></p>
    </div>
    <div class="tag"><b>Track 2 — Quantitative Diversity Measurement</b><br>
      team rvnikita &#183; EgoVerse, NYC &#183; 15 Aug 2026<br>
      {A['n_episodes']} real episodes &#183; DINOv2 on a Modal L4</div>
  </div>

  <div class="lanes">
    <div class="lane">
      <div class="who" style="color:#8f8e86">Today<span>LLM as a judge</span></div>
      <div class="chain">
        <div class="node">clips<b>a subset</b></div>{arw}
        <div class="node">prompt<b>text</b></div>{arw}
        <div class="node cold">{J['model']}<b>API call</b></div>{arw}
        <div class="node">&ldquo;{J['diverse']['scores'][0]:.0f}?&rdquo;<b>a guess</b></div>
        <div class="loop">&#8630; pay again for<br>every new question</div>
      </div>
      <div class="cost"><span class="big bad">${per:.4f}</span>
        per question &#183; <span class="bad">{d0['spread']:.0f}-pt spread at T=0</span><br>
        30% duplicates &#8594; scores <span class="bad">higher</span>
        ({J['diverse']['mean']:.0f}&#8594;{J['diverse_30pct_duplicated']['mean']:.0f})</div>
    </div>

    <div class="lane ours">
      <div class="who" style="color:#d95926">Ours<span>embedding index</span></div>
      <div class="chain">
        <div class="node">32 frames<b>uniform</b></div>{arw}
        <div class="node hot">DINOv2<b>Modal L4</b></div>{arw}
        <div class="node">1 vector<b>per episode</b></div>{arw}
        <div class="node">index<b>built once</b></div>{arw}
        <div class="node">Vendi score<b>numpy</b></div>
        <div class="fan"><i>&#8594; score a subset</i><i>&#8594; rank two subsets</i>
          <i>&#8594; &#8734; more, free</i></div>
      </div>
      <div class="cost"><span class="big good">${E['index_cost_usd']:.3f} once</span>
        then <b class="good">{n_q:,} scores free</b> &#183; identical every run<br>
        30% duplicates &#8594; <span class="good">provably cannot</span> raise it
        ({A['duplication']['exact_clones']['delta']:+.3f})</div>
    </div>
  </div>

  <div class="main">
    <div class="viz">
      <div class="vizh">
        <div class="vizt">every episode, coloured by which pick represents it</div>
        <div class="leg"><span class="dot" style="background:#d95926"></span>diverse only
          <span class="dot" style="background:#3987e5"></span>random only
          <span class="dot" style="background:#7d7c74"></span>both
          <span class="dot" style="background:#3a3934"></span>neither</div>
      </div>
      <canvas id="cloud"></canvas>
      <div class="vizf">{excl_d} episodes reached <b style="color:#d95926">only</b> by the
        diverse pick vs {excl_r} reached only by random &#8212;
        <b style="color:#fff">{excl_d/max(excl_r,1):.1f}&#215; more exclusive coverage</b>
        for the same budget of 32</div>
    </div>

    <div class="panel">
      {row('Random 32', R, '#3987e5', '', f"avg distance {R['mean_dist_to_pick']:.3f}")}
      {row('Diverse 32', Dv, '#d95926', 'win',
           f"avg distance {Dv['mean_dist_to_pick']:.3f} &#183; "
           f"<b>{(1-Dv['mean_dist_to_pick']/R['mean_dist_to_pick'])*100:.0f}% closer</b>")}
      <div class="warn"><b>The score alone is not the goal.</b> Pure farthest-point scores
        {Sp['vendi']:.2f} — far higher — but collects outliers and represents only
        {Sp['covered_pct']:.0f}% of the corpus. We select by cluster cover, which wins on
        <em>both</em> the score and coverage. Measured, in the repo.</div>
    </div>
  </div>

  <div class="foot">
    <div><div class="fl">Broader coverage</div>
      <div class="ft"><b>{cov['recording days']['by']['diverse']['mean'][2]:.1f} of
        {cov['recording days']['total']}</b> recording sessions at k=16 vs
        {cov['recording days']['by']['random']['mean'][2]:.1f} —
        metadata the encoder never saw.</div></div>
    <div><div class="fl">Edge cases on purpose</div>
      <div class="ft">A dial on how far from typical you sample — rare setups get
        <b>bought, not stumbled on</b>.</div></div>
    <div><div class="fl">Failures surface early</div>
      <div class="ft">Failed demos look unlike successes, so they rise in the same ranking:
        <b>{enr['top32_enrichment']}&#215; base rate</b>, no labels.</div></div>
    <div class="qr">{qr_svg(REPO)}
      <div><div class="repo">github.com/rvnikita/<br>egoverse-diversity</div>
        <div class="repos"><b style="color:#c3c2b7">python run_all.py</b><br>
          every number here &#183; 6 s &#183; no GPU, no keys</div></div>
    </div>
  </div>

</div></div>
<script>{JS % {"data": data}}</script>
</body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
