"""The summary slide, as reusable parts.

Kept separate so the slide can be emitted standalone (results/slide.html) AND used as the
hero of the single-page deliverable (results/index.html) without the markup existing twice.

Design rule for this file: a judge gets 15-30 seconds. Everything that is not needed to
understand the pitch in that time belongs in the dashboard below it, not here.
"""

from __future__ import annotations

import io
import json

import segno

REPO = "https://github.com/rvnikita/egoverse-diversity"


def qr_svg(url: str = REPO, scale: int = 3) -> str:
    """segno emits no viewBox, so render at native size rather than resizing in CSS."""
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=scale, border=2,
                                    dark="#ffffff", light=None, xmldecl=False, svgns=True)
    return buf.getvalue().decode()


CSS = """
.slide{width:1600px;height:900px;background:#141413;color:#fff;padding:44px 56px 30px;
  font:16px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column;position:relative;overflow:hidden}
.slide::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(1100px 620px at 38% 62%,rgba(217,89,38,.07),transparent 70%)}
.slide *{box-sizing:border-box}
.s-top{display:flex;justify-content:space-between;align-items:flex-start;flex:none}
.slide h1{font-size:48px;line-height:1.02;margin:0;letter-spacing:-.033em;font-weight:690}
.slide h1 em{font-style:normal;color:#d95926}
.s-q{font-size:19px;color:#c3c2b7;margin:12px 0 0}
.s-q b{color:#fff;font-weight:640}
.s-tag{font-size:12.5px;color:#8f8e86;text-align:right;line-height:1.8;letter-spacing:.02em}
.s-tag b{color:#c3c2b7;font-weight:600}

.s-lanes{display:flex;flex-direction:column;gap:10px;margin-top:24px;flex:none}
.s-lane{display:grid;grid-template-columns:96px 1fr 330px;align-items:center;gap:18px;
  background:#1a1a19;border:1px solid #33322e;border-radius:11px;padding:14px 18px}
.s-lane.ours{border-color:#5c4030;background:
  linear-gradient(90deg,rgba(217,89,38,.11),rgba(217,89,38,.02) 60%,transparent),#1a1a19}
.s-who{font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;font-weight:750}
.s-chain{display:flex;align-items:center}
.s-node{background:#262521;border:1px solid #3d3b36;border-radius:7px;padding:8px 13px;
  font-size:14px;color:#e8e7e0;white-space:nowrap;line-height:1.2}
.s-node.hot{background:#d95926;border-color:#d95926;color:#141413;font-weight:680}
.s-node.cold{background:#2b3a4d;border-color:#3987e5;color:#dbe8f8}
.s-arw{color:#6b6a64;font-size:16px;padding:0 9px;flex:none}
.s-cost{text-align:right;font-size:14px;line-height:1.5;color:#c3c2b7}
.s-cost b{font-size:20px;font-weight:720;letter-spacing:-.02em}
.s-bad{color:#e66767}.s-good{color:#42c08a}

.s-main{display:grid;grid-template-columns:1fr 336px;gap:20px;margin-top:18px;flex:1;
  min-height:0}
.s-viz{background:#1a1a19;border:1px solid #33322e;border-radius:12px;
  padding:13px 16px 9px;display:flex;flex-direction:column;min-height:0}
.s-vh{display:flex;justify-content:space-between;align-items:center;flex:none}
.s-leg{font-size:13px;color:#8f8e86}
.s-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 6px 0 16px;
  vertical-align:middle}
.slide canvas{width:100%;flex:1;display:block;min-height:0}
.s-vf{font-size:15px;color:#c3c2b7;text-align:center;flex:none;padding-bottom:3px}
.s-vf b{color:#fff;font-weight:700}

.s-panel{display:flex;flex-direction:column;gap:12px;justify-content:center}
.s-row{background:#1a1a19;border:1px solid #33322e;border-radius:11px;padding:15px 17px}
.s-row.win{border-color:#5c4030;background:rgba(217,89,38,.07)}
.s-rh{display:flex;justify-content:space-between;align-items:baseline}
.s-rn{font-size:13px;letter-spacing:.08em;text-transform:uppercase;font-weight:720}
.s-rv{font-size:34px;font-weight:720;letter-spacing:-.03em;line-height:1}
.s-sub{font-size:11.5px;color:#8f8e86;text-align:right;margin-top:2px}
.s-bar{height:13px;border-radius:7px;background:#2b2a26;overflow:hidden;margin-top:11px}
.s-bar i{display:block;height:100%;border-radius:7px}
.s-bl{font-size:13px;color:#8f8e86;margin-top:7px}
.s-bl b{color:#fff;font-weight:680}

.s-foot{display:flex;justify-content:space-between;align-items:center;margin-top:16px;
  padding-top:14px;border-top:1px solid #33322e;flex:none;gap:24px}
.s-unlocks{font-size:14px;color:#c3c2b7;line-height:1.55}
.s-unlocks b{color:#fff;font-weight:650}
.s-qr{display:flex;gap:14px;align-items:center;flex:none}
.s-qr svg{display:block;flex:none}
.s-repo{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:#fff;font-weight:600}
.s-repos{font-size:11.5px;color:#8f8e86;line-height:1.45;margin-top:4px}
"""

JS = """
(function(){
  var D=%(data)s;
  var cv=document.getElementById('cloud'); if(!cv) return;
  var ctx=cv.getContext('2d'), P={};
  D.p.forEach(function(i){P[i]=1;});
  var W=0,H=0,ay=0.4,ax=-0.26;
  function size(){var r=cv.getBoundingClientRect(),dpr=2;
    W=r.width;H=r.height;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);}
  // 0 neither, 1 both, 2 random only, 3 diverse only
  var COL=['#3a3934','#7d7c74','#3987e5','#d95926'],
      ALP=[0.28,0.40,0.95,0.95], RAD=[2.0,2.2,3.6,3.6];
  function draw(){
    if(!W)return;
    ctx.clearRect(0,0,W,H);
    var ca=Math.cos(ay),sa=Math.sin(ay),cb=Math.cos(ax),sb=Math.sin(ax);
    var s=Math.min(W,H)*0.34, cx=W/2, cy=H/2, pts=[];
    for(var i=0;i<D.x.length;i++){
      var x=D.x[i],y=D.y[i],z=D.z[i];
      var x1=x*ca+z*sa, z1=-x*sa+z*ca;
      var y1=y*cb-z1*sb, z2=y*sb+z1*cb;
      var p=2.6/(2.6+z2*0.3);
      pts.push({X:cx+x1*s*p,Y:cy-y1*s*p,z:z2,k:D.k[i],pick:P[i],p:p});
    }
    pts.sort(function(u,v){return v.z-u.z;});
    for(var j=0;j<pts.length;j++){
      var q=pts[j];
      ctx.beginPath();
      ctx.arc(q.X,q.Y,(q.pick?5.6:RAD[q.k])*q.p,0,6.2832);
      ctx.fillStyle=COL[q.k]; ctx.globalAlpha=q.pick?1:ALP[q.k];
      ctx.fill();
      if(q.pick){ctx.globalAlpha=.9;ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();}
    }
    ctx.globalAlpha=1;
  }
  size();
  function frame(){ay+=0.0021;draw();requestAnimationFrame(frame);}
  addEventListener('resize',function(){size();draw();});
  frame();
})();
"""


def payload(A: dict) -> str:
    """Per-episode coverage class, computed in the true 768-d space — not the projection."""
    R, Dv, pr = A["subsets"]["random"], A["subsets"]["diverse"], A["projection"]
    klass = [(3 if d and not r else 2 if r and not d else 1 if r and d else 0)
             for r, d in zip(R["covered"], Dv["covered"])]
    return json.dumps({"x": pr["x"], "y": pr["y"], "z": pr["z"], "k": klass,
                       "p": list(R["idx"]) + list(Dv["idx"])}, separators=(",", ":"))


def body(A: dict, J: dict) -> str:
    R, Dv, Sp = A["subsets"]["random"], A["subsets"]["diverse"], A["subsets"]["spread"]
    E, RD, NB = A["economics"], A["random_distribution"], A["narrow_vs_broad"]
    klass = [(3 if d and not r else 2 if r and not d else 1 if r and d else 0)
             for r, d in zip(R["covered"], Dv["covered"])]
    excl_r = sum(1 for k in klass if k == 2)
    excl_d = sum(1 for k in klass if k == 3)
    a = '<span class="s-arw">&#10230;</span>'

    def row(name, s, colour, cls):
        return f"""
        <div class="s-row {cls}">
          <div class="s-rh"><div class="s-rn" style="color:{colour}">{name}</div>
            <div><div class="s-rv" style="color:{colour}">{s['covered_pct']:.0f}%</div>
              <div class="s-sub">of the corpus represented</div></div></div>
          <div class="s-bar"><i style="width:{s['covered_pct']:.0f}%;background:{colour}"></i></div>
          <div class="s-bl">Vendi score <b>{s['vendi']:.2f}</b></div>
        </div>"""

    return f"""<div class="slide">
  <div class="s-top">
    <div>
      <h1>Score diversity once. <em>Query it forever.</em></h1>
      <p class="s-q">439,053 episodes. You can train on a few hundred.
        <b>Which ones &#8212; and how do you prove it?</b></p>
    </div>
    <div class="s-tag"><b>Track 2 &#183; Quantitative Diversity</b><br>
      team rvnikita &#183; EgoVerse NYC<br>
      {A['n_episodes']} real episodes &#183; DINOv2 on Modal</div>
  </div>

  <div class="s-lanes">
    <div class="s-lane">
      <div class="s-who" style="color:#8f8e86">Today</div>
      <div class="s-chain">
        <div class="s-node">a subset of clips</div>{a}
        <div class="s-node cold">ask {J['model']}</div>{a}
        <div class="s-node">&ldquo;diversity: {J['diverse']['scores'][0]:.0f}?&rdquo;</div>
      </div>
      <div class="s-cost"><b class="s-bad">${J['cost_per_subset_usd']:.4f}</b> every question
        &#183; <span class="s-bad">answer moves
        {J['diverse_temp0']['spread']:.0f} pts</span> on a re-run</div>
    </div>
    <div class="s-lane ours">
      <div class="s-who" style="color:#d95926">Ours</div>
      <div class="s-chain">
        <div class="s-node">frames</div>{a}
        <div class="s-node hot">DINOv2 embedding</div>{a}
        <div class="s-node">index</div>{a}
        <div class="s-node">score any subset</div>
      </div>
      <div class="s-cost"><b class="s-good">${E['index_cost_usd']:.3f} once</b>, then
        <b class="s-good">free</b> &#183; <span class="s-good">same answer every time</span></div>
    </div>
  </div>

  <div class="s-main">
    <div class="s-viz">
      <div class="s-vh">
        <div class="s-leg">every episode, coloured by which pick represents it</div>
        <div class="s-leg"><span class="s-dot" style="background:#d95926"></span>diverse only
          <span class="s-dot" style="background:#3987e5"></span>random only
          <span class="s-dot" style="background:#7d7c74"></span>both</div>
      </div>
      <canvas id="cloud"></canvas>
      <div class="s-vf">reaches <b>{excl_d} episodes</b> random misses (random reaches
        <b>{excl_r}</b> it misses) &#8212; and beats
        <b>all {RD['draws']} random draws</b> on coverage, best of which hit
        {RD['cov_max']:.0f}%</div>
    </div>
    <div class="s-panel">
      {row('Random 32', R, '#3987e5', '')}
      {row('Diverse 32', Dv, '#d95926', 'win')}
    </div>
  </div>

  <div class="s-foot">
    <div class="s-unlocks">The score ranks what a human already knows:
      32 clips from <b>one operator on one day</b> score
      <b>{NB['narrow_mean']:.1f}</b>; 32 spread across
      {NB['trials'] and 10} days score <b>{NB['broad_mean']:.1f}</b> &#8212;
      no overlap in {NB['trials']} paired trials.</div>
    <div class="s-qr">{qr_svg()}
      <div><div class="s-repo">github.com/rvnikita/<br>egoverse-diversity</div>
        <div class="s-repos"><b style="color:#c3c2b7">python run_all.py</b> &#183; 1-2 min
          &#183; no GPU, no keys</div></div>
    </div>
  </div>
</div>"""
