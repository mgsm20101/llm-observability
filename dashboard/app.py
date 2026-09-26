"""FastAPI dashboard over the local trace sink — read-only.

    uvicorn dashboard.app:app --port 8080
    http://127.0.0.1:8080

Reads `runs/traces.jsonl`, the file `run_traced.py` and `eval_ci` append to.
Per-stage figures come from `src/stats.stage_summary()`, the same function
that writes the stages block of `results/summary_<sha8>.json`, so a stage's
share is computed the same way on both surfaces. The dashboard covers every
run in the sink; a summary covers one run.

**The headline here is the stage breakdown, not the average.** An average
end-to-end latency tells you a request was slow; `generate` at 70% of wall clock
tells you which of three things to go and fix.
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.stats import root_durations, stage_summary
from src.tracer import TRACE_PATH, read_traces

app = FastAPI(title="LLM Observability Dashboard", version="0.2.0")


@app.get("/metrics")
async def get_metrics() -> dict:
    """Aggregate every span currently in the sink."""
    rows = read_traces()
    if not rows:
        return {
            "message": "No traces yet. Run `python run_traced.py` (or "
                       f"`python -m src.eval_ci`), then reload. Sink: {TRACE_PATH}"
        }

    spans = [r for r in rows if r.get("kind") != "score"]
    scores = [r for r in rows if r.get("kind") == "score"]

    stages = stage_summary(spans)

    # The sink stores usage as {"input": n, "output": n} — there is no
    # total_tokens key and no cost on the span. Cost is modelled downstream in
    # src/cost.py from a price table, so it is deliberately not reported here:
    # a dashboard tile is the wrong place for a number the traces do not hold.
    tokens_in = sum(int((s.get("usage") or {}).get("input") or 0) for s in spans)
    tokens_out = sum(int((s.get("usage") or {}).get("output") or 0) for s in spans)

    by_score: dict[str, list[float]] = defaultdict(list)
    for row in scores:
        by_score[row["name"]].append(float(row["value"]))

    return {
        "sink": str(TRACE_PATH),
        "trace_count": len(root_durations(spans)),
        "span_count": len(spans),
        "stages": stages,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "scores": {
            name: {"n": len(v), "mean": round(sum(v) / len(v), 3)}
            for name, v in sorted(by_score.items())
        },
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui() -> str:
    return """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Observability — P7</title>
<style>
  :root{--primary:#1a56db;--surface:#f9fafb;--card:#fff;--border:#e5e7eb;
        --text-main:#111827;--text-muted:#6b7280;}
  body{font-family:"Segoe UI",system-ui,sans-serif;background:var(--surface);
       color:var(--text-main);padding:24px;max-width:860px;margin:0 auto;}
  h1{color:#1e3a8a;border-bottom:3px solid var(--primary);padding-bottom:8px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;
        padding:20px;margin:16px 0;}
  .metric{display:inline-block;margin:8px 24px 8px 0;}
  .metric .val{font-size:22pt;font-weight:700;color:var(--primary);}
  .metric .label{font-size:10pt;color:var(--text-muted);}
  table{width:100%;border-collapse:collapse;margin-top:8px;}
  th,td{padding:8px;border-bottom:1px solid var(--border);text-align:right;}
  th{font-size:10pt;color:var(--text-muted);font-weight:600;}
  .bar{height:8px;background:var(--primary);border-radius:4px;display:inline-block;}
  .muted{color:var(--text-muted);font-size:10pt;}
  button{background:var(--primary);color:#fff;border:none;border-radius:8px;
         padding:8px 18px;cursor:pointer;font-size:11pt;}
</style>
</head>
<body>
<h1>LLM Observability — P7</h1>
<p class="muted">يقرأ مباشرةً من <code>runs/traces.jsonl</code> — الملف الذي يكتب فيه
<code>run_traced.py</code> وبوابة الـ CI. أرقام المراحل محسوبة بنفس دوال
<code>src/stats.py</code>.</p>

<div class="card">
  <h3>الإجمالي</h3>
  <div id="totals">…</div>
  <button onclick="load()">تحديث</button>
</div>

<div class="card">
  <h3>الزمن لكل مرحلة</h3>
  <p class="muted">النصيب من زمن الـ pipeline هو المخرج المهم — المتوسط الكلي
  يقول إن الطلب بطيء، وهذا الجدول يقول أين.</p>
  <div id="stages">…</div>
</div>

<div class="card">
  <h3>الدرجات المسجَّلة على الـ traces</h3>
  <div id="scores">…</div>
</div>

<script>
async function load() {
  const d = await (await fetch('/metrics')).json();
  if (d.message) {
    document.getElementById('totals').innerHTML = '<p>' + d.message + '</p>';
    return;
  }
  document.getElementById('totals').innerHTML = `
    <div class="metric"><div class="val">${d.trace_count}</div><div class="label">Traces</div></div>
    <div class="metric"><div class="val">${d.span_count}</div><div class="label">Spans</div></div>
    <div class="metric"><div class="val">${d.total_tokens}</div><div class="label">Tokens (${d.tokens_in} in / ${d.tokens_out} out)</div></div>`;

  const rows = Object.entries(d.stages).map(([name, s]) => `
    <tr><td>${name}</td><td>${s.n}</td><td>${s.median_ms}</td><td>${s.p95_ms}</td><td>${s.max_ms}</td>
    <td>${s.share_of_total === null ? '—'
        : s.share_of_total + '% <span class="bar" style="width:'
          + s.share_of_total + 'px"></span>'}</td></tr>`).join('');
  document.getElementById('stages').innerHTML =
    `<table><tr><th>المرحلة</th><th>n</th><th>median ms</th><th>p95 ms</th><th>max ms (بارد)</th>
     <th>النصيب</th></tr>${rows}</table>`;

  const sc = Object.entries(d.scores).map(([name, s]) =>
    `<tr><td>${name}</td><td>${s.n}</td><td>${s.mean}</td></tr>`).join('');
  document.getElementById('scores').innerHTML = sc
    ? `<table><tr><th>الدرجة</th><th>n</th><th>المتوسط</th></tr>${sc}</table>`
    : '<p class="muted">لا توجد درجات بعد.</p>';
}
load();
</script>
</body>
</html>"""
