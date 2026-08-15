"""Emit the standalone summary slide, results/slide.html.

The markup lives in scripts/slide_parts.py so the same slide can be the hero of the
single-page deliverable (results/index.html) without existing twice.

    .venv/bin/python scripts/build_slide.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import slide_parts as SP  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "slide.html"


def main() -> int:
    A = json.loads((RESULTS / "analysis.json").read_text())
    J = json.loads((RESULTS / "llm_judge.json").read_text())

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diversity Index — slide</title>
<style>
html,body{{margin:0;background:#0a0a09;overflow:hidden;height:100%}}
#fit{{height:100%;display:grid;place-items:center}}
#stage{{transform-origin:center center}}
{SP.CSS}
</style></head><body>
<div id="fit"><div id="stage">{SP.body(A, J)}</div></div>
<script>
(function(){{
  var st=document.getElementById('stage');
  function fit(){{st.style.transform='scale('+Math.min(innerWidth/1600,innerHeight/900)+')';}}
  addEventListener('resize',fit); fit();
}})();
{SP.JS % {"data": SP.payload(A)}}
</script>
</body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
