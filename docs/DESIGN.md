# Design — 07-llm-observability

The problem, the architecture and the file map are in the
[README](../README.md#structure); results are in [results.md](results.md).
This file records only the decisions and what each one costs.

## Key design decisions

### Why a local JSONL sink and not Langfuse Cloud

The original design called for Langfuse's free tier, and it was never run: the
package was not installed, two API keys were required, and the retrieval half
pointed at a Qdrant instance that was not running. Three external dependencies
for a project whose entire subject is measuring things locally.

`src/tracer.py` replaces it with a handful of plain functions — `observe`,
`current_trace_id`, `annotate`, `add_score`, `read_traces`. A tracing SDK is a
context variable holding the current span, a stack discipline for nesting, and
a writer. Making that explicit is worth more here than the dashboard it gave
up.

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

### Why one stats module and one eval loop

`eval_ci` took `sorted(v)[n // 2]` and the dashboard took a nearest-rank
percentile, so the same set of traces produced two different `generate`
medians for identical data. Two components of one project disagreeing about
identical data is a bug that looks like a measurement. Every latency figure now
comes from `src/stats.py`; the dashboard and the summary JSON both call
`stage_summary()`, including its `n × median / root total` share.

The same reasoning applies to scoring: `run_traced.py` used to carry its own
copy of the eval loop. Both entry points now call `eval_ci.score_cases()` and
`eval_ci.gate_failures()`, so a change to how an answer is scored or gated
cannot reach one entry point and miss the other.

### Why cost.py is a pure module

No I/O, so it is unit-testable without mocking a client, and so the distinction
stays visible: token counts are measured, prices are a table, and a cost is a
multiplication of the two. Local inference is billed at zero; any hosted figure
is labelled a projection wherever it appears.
