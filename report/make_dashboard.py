#!/usr/bin/env python3
"""One self-contained HTML file, built from the same results/*.jsonl.

    python report/make_dashboard.py

The dashboard is a **view, never a source**. It computes nothing the static
figures do not -- same loader, same specs, same numbers. `tests/test_figures.py`
asserts both read identical keys, so the two cannot drift apart and quietly
tell different stories.

No external requests: data is embedded at build time and the chart is drawn with
inline SVG, so the file works offline and from a `file://` URL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_figures import SPECS, load  # noqa: E402

OUT = ROOT / "report" / "dashboard.html"


def collect() -> dict[str, list[dict]]:
    """Every results file any figure declares, loaded once."""
    return {spec.source: load(spec.source) for spec in SPECS.values()}


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>llm-inference-stack — results</title>
<style>
  :root {
    --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2;
    --problem:#c44e52; --fix:#4c72b0; --accent:#55a868; --third:#dd8452;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16181c; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2c2f36; }
  }
  body { background:var(--bg); color:var(--fg); margin:0; padding:2rem 1.25rem;
         font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:960px; margin:0 auto; }
  h1 { font-size:1.4rem; margin:0 0 .25rem; }
  .sub { color:var(--muted); margin:0 0 2rem; }
  section { border-top:1px solid var(--line); padding:1.5rem 0; }
  h2 { font-size:1rem; margin:0 0 .2rem; }
  .note { color:var(--muted); font-size:.85rem; margin:0 0 1rem; }
  .empty { color:var(--muted); font-style:italic; }
  .wrap { overflow-x:auto; }
  table { border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:.85rem; }
  th,td { padding:.35rem .7rem; text-align:right; border-bottom:1px solid var(--line); }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; }
  .gate { color:var(--accent); }
  .miss { color:var(--problem); }
  label { margin-right:1rem; font-size:.85rem; color:var(--muted); }
</style>
<main>
  <h1>llm-inference-stack — results</h1>
  <p class="sub">Generated from <code>results/*.jsonl</code>. A view, not a source:
     every number here is computed by the same functions that draw the static figures.</p>
  <div id="app"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const MFU_GATE = __MFU_GATE__;
const COLOR = {unfused:"var(--problem)", torch_flash:"var(--fix)", triton:"var(--third)"};
const app = document.getElementById("app");

function section(title, note, body) {
  const s = document.createElement("section");
  s.innerHTML = `<h2>${title}</h2><p class="note">${note}</p>`;
  s.appendChild(body);
  app.appendChild(s);
}

function empty(msg) {
  const p = document.createElement("p");
  p.className = "empty"; p.textContent = msg;
  return p;
}

/* Log-x scatter/line chart. Deliberately minimal -- the static PNGs are the
   publication artifact; this exists to let you filter and hover. */
function chart(rows, xk, yk, yfmt, ylabel) {
  const W=880,H=320,P={t:14,r:14,b:42,l:58};
  const xs=rows.map(r=>Math.log2(r[xk])), ys=rows.map(r=>r[yk]);
  const x0=Math.min(...xs), x1=Math.max(...xs)||x0+1;
  const y1=Math.max(...ys, MFU_GATE*100)*1.15;
  const px=v=>P.l+(Math.log2(v)-x0)/((x1-x0)||1)*(W-P.l-P.r);
  const py=v=>H-P.b-(v/y1)*(H-P.t-P.b);

  let g=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  g+=`<line x1="${P.l}" y1="${H-P.b}" x2="${W-P.r}" y2="${H-P.b}" stroke="var(--line)"/>`;
  g+=`<line x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${H-P.b}" stroke="var(--line)"/>`;
  for(let i=0;i<=4;i++){const v=y1*i/4;
    g+=`<line x1="${P.l}" y1="${py(v)}" x2="${W-P.r}" y2="${py(v)}" stroke="var(--line)" opacity=".5"/>`;
    g+=`<text x="${P.l-8}" y="${py(v)+4}" text-anchor="end" font-size="10" fill="var(--muted)">${v.toFixed(1)}</text>`;}
  g+=`<line x1="${P.l}" y1="${py(MFU_GATE*100)}" x2="${W-P.r}" y2="${py(MFU_GATE*100)}"
        stroke="var(--accent)" stroke-dasharray="5 4"/>`;
  g+=`<text x="${W-P.r}" y="${py(MFU_GATE*100)-6}" text-anchor="end" font-size="10"
        fill="var(--accent)">gate ${(MFU_GATE*100).toFixed(0)}%</text>`;

  const by={}; rows.forEach(r=>(by[r.backend]??=[]).push(r));
  for(const [b,rs] of Object.entries(by)){
    rs.sort((a,c)=>a[xk]-c[xk]);
    const c=COLOR[b]||"var(--muted)";
    g+=`<polyline fill="none" stroke="${c}" stroke-width="2" points="${
      rs.map(r=>`${px(r[xk])},${py(r[yk])}`).join(" ")}"/>`;
    rs.forEach(r=>{g+=`<circle cx="${px(r[xk])}" cy="${py(r[yk])}" r="4" fill="${c}">
      <title>${b} · ${xk}=${r[xk]} · ${yfmt(r[yk])}</title></circle>`;});
  }
  [...new Set(rows.map(r=>r[xk]))].sort((a,b)=>a-b).forEach(v=>{
    g+=`<text x="${px(v)}" y="${H-P.b+16}" text-anchor="middle" font-size="10"
          fill="var(--muted)">${v}</text>`;});
  g+=`<text x="${W/2}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${xk}</text>`;
  g+=`<text transform="translate(14,${H/2}) rotate(-90)" text-anchor="middle" font-size="11"
        fill="var(--muted)">${ylabel}</text></svg>`;

  let legend=Object.keys(by).map(b=>
    `<label><span style="color:${COLOR[b]||"var(--muted)"}">■</span> ${b}</label>`).join("");
  const d=document.createElement("div");
  d.className="wrap"; d.innerHTML=legend+g;
  return d;
}

function table(rows, cols) {
  const d=document.createElement("div"); d.className="wrap";
  d.innerHTML = `<table><thead><tr>${cols.map(c=>`<th>${c[0]}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${c[1](r)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  return d;
}

const kernel = DATA["kernel.jsonl"] || [];
if (!kernel.length) {
  section("Kernel", "MFU and roofline per backend.",
    empty("No results/kernel.jsonl. Run bench/kernel.py, or copy results/ back from a GPU box."));
} else {
  section("K1 — achieved utilization",
    "Fraction of dense peak reached. Below the gate, fusing bought nothing.",
    chart(kernel, "seq", "mfu_pct", v=>v.toFixed(2)+"%", "MFU (%)"));
  section("K2 — where the time goes",
    "Low arithmetic intensity means bandwidth-bound: the kernel is waiting on HBM, not on math.",
    table(kernel, [
      ["backend", r=>r.backend],
      ["seq", r=>r.seq],
      ["TFLOP/s", r=>r.tflops.toFixed(2)],
      ["MFU", r=>`<span class="${r.mfu>=MFU_GATE?"gate":"miss"}">${(r.mfu*100).toFixed(2)}%</span>`],
      ["FLOP/byte", r=>r.arithmetic_intensity.toFixed(1)],
      ["ms", r=>r.ms.toFixed(3)],
    ]));
}
</script>
"""


def main() -> None:
    from lis.metrics import MFU_GATE

    data = collect()
    for rows in data.values():
        for r in rows:
            if "mfu" in r:
                r["mfu_pct"] = r["mfu"] * 100

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__MFU_GATE__", repr(MFU_GATE)))
    OUT.write_text(html)

    n = sum(len(v) for v in data.values())
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KiB, {n} records)")
    if not n:
        print("  no data embedded — run bench/kernel.py first")


if __name__ == "__main__":
    main()
