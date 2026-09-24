# Design — 07-llm-observability

## Problem

RAG pipelines fail silently in production:

- the model hallucinates and the request returns HTTP 200
- a prompt change breaks three eval cases and nobody notices
- one request burns 5× the expected tokens
- latency spikes with no visibility into *which* step got slow

`print()` logging gives prose. What you need is structured, queryable,
per-request data you can aggregate and gate on.

## Architecture

```
rag_observed.answer_question()   ← @observe("rag_pipeline")  → trace root
    │
    ├─── _retrieve()             ← @observe("retrieve")      → span
    │        └─ in-process dense store, intfloat/multilingual-e5-base
    │
    ├─── _rerank()               ← @observe("rerank")        → span
    │        └─ cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, top 3
    │
    └─── _generate()             ← @observe(..., as_type="generation")
             └─ gemma3:4b via Ollama, token usage read back from the response

Every span → runs/traces.jsonl, one JSON object per line, flushed on close.

Eval CI (src/eval_ci.py):
    ├─ 12 cases: 10 answerable, 2 deliberately not
    ├─ three deterministic checks — retrieval / grounding / abstention
    ├─ writes each verdict back onto the trace as a score row
    └─ sys.exit(1) if any pre-registered threshold breaks
```

## Key design decisions

### Why a local JSONL sink and not Langfuse Cloud

The original design called for Langfuse's free tier, and it was never run: the
package was not installed, two API keys were required, and the retrieval half
pointed at a Qdrant instance that was not running. Three external dependencies
for a project whose entire subject is measuring things locally.

`src/tracer.py` replaces it in ~150 lines with the same public surface the call
sites already used (`observe`, `langfuse_context`, `add_score`, `flush`), so
`rag_observed.py` and `eval_ci.py` did not have to be rebuilt around a different
idea. A tracing SDK is a context variable holding the current span, a stack
discipline for nesting, and a writer. Making that explicit is worth more here
than the dashboard it gave up.

What this costs: nothing here is evidence of integrating with a hosted backend,
and the README says so.

### Why the span rows are written immediately, not batched

A hosted client batches because each flush is a network round trip. A local file
has no such reason, and buffering would lose the trace of the run that crashed —
the one run whose trace matters most.

### Why deterministic scoring and not an LLM judge

An earlier version of this eval scored faithfulness and relevance with the
same model that generated the answer, and the judge was never validated
against human labels. That produces a number nobody can defend: when the judge
is wrong there is no way to find out, and `faithfulness 0.82` reads like a
measurement while being an opinion with a decimal point.

All three checks now resolve against the corpus or against a string the eval set
fixed in advance, so every verdict replays exactly from the saved traces.

### Why abstention and false abstention are reported as a pair

A system that refused every question scores 1.000 on abstention and is useless.
Neither number means anything alone, so `eval_ci` prints them adjacent and the
README repeats the warning.

### Why one shared `median()`

`eval_ci` took `sorted(v)[n // 2]` and the dashboard took a nearest-rank
percentile, so the same set of traces produced two different `generate`
medians for identical data. Two components of one project disagreeing about
identical data is a bug that looks like a measurement. Both now call
`tracer.median()`, which averages the middle pair at even n.

### Why cost.py is a pure module

No I/O, so it is unit-testable without mocking a client, and so the distinction
stays visible: token counts are measured, prices are a table, and a cost is a
multiplication of the two. Local inference is billed at zero; any hosted figure
is labelled a projection wherever it appears.

## Results

Written by `run_traced.py`, not by hand: see [`results.md`](results.md) and
`results/summary_<sha8>.json`. Numbers there are tied to a commit SHA and a
clean worktree at run time; nothing here restates a figure that is not backed
by a result file in the tree.
