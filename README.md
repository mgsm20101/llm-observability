# LLM Observability — traces, a stage breakdown, and a CI gate that can fail

A RAG pipeline that reports only end-to-end latency tells you a request was
slow. This one tells you **which of three stages** was slow, on every request,
and fails the build when quality drops below a threshold fixed before the run.

## Problem

RAG pipelines fail quietly. A hallucination returns HTTP 200. A prompt tweak
breaks three eval cases and nobody notices until a user does. One request burns
5× the expected tokens and the log line says `answered ok`. Ordinary logging
gives you prose; what you need is structured per-request data you can aggregate.

## Architecture

```
answer_question(q)
  └─ rag_pipeline ──────────────── trace root
       ├─ retrieve    dense search over 12 docs, intfloat/multilingual-e5-base
       ├─ rerank      cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 → top 3
       └─ generate    gemma3:4b via Ollama, real token counts read back

Every span → runs/traces.jsonl (one JSON object per line).
```

**The sink is a local file, not a hosted service.** `src/tracer.py` is ~150
lines: a context variable holding the current span, a stack discipline for
nesting, and a writer. It keeps the public surface the call sites already used
(`observe`, `langfuse_context`, `add_score`, `flush`), so the pipeline did not
have to be rebuilt around a different idea.

That was not a stylistic choice. The previous version imported `langfuse` (not
installed), required two API keys for a cloud account, and queried a Qdrant
instance that was not running — three external dependencies for a project whose
entire subject is measuring things locally. It could not be run at all, and
`docs/results.md` was empty.

## Run

```bash
pip install -r requirements.txt
ollama serve && ollama pull gemma3:4b

python -m src.eval_ci                      # exit 0 = every gate held
uvicorn dashboard.app:app --port 8080      # http://127.0.0.1:8080
```

No keys, no network, no vector-DB container.

## Eval

Twelve cases in `data/eval_set.jsonl` — 10 answerable from the 12-document HR
corpus, 2 deliberately not. Three checks, thresholds fixed **before** the run:

| Check | Threshold |
|---|---|
| retrieval — the document holding the answer came back at all | ≥ 0.80 |
| grounding — the answer contains the fact the eval set requires | ≥ 0.70 |
| abstention — on an unanswerable question, it declined | ≥ 0.50 |

**Scored by a program, not by a judge.** An earlier version of this gate scored
answers with a second LLM call judging faithfulness and relevance — a model
scoring another model's output, never validated against human labels. That
produces a number nobody can defend: when the judge is wrong there is no way to
find out, and "faithfulness 0.82" reads like a measurement while being an
opinion with a decimal point. Every check now resolves against the corpus or
against a string the eval set fixed in advance, so verdicts replay exactly from
the saved traces.

## Results

Measured at commit `b382cd4e` on a clean tree, `gemma3:4b` through local Ollama —
raw files [`results/traces_b382cd4e.jsonl`](results/traces_b382cd4e.jsonl) (every span) and
[`results/summary_b382cd4e.json`](results/summary_b382cd4e.json) (the summary `run_traced.py` wrote).
12 questions, 48 spans. Warm figures and warm shares below are
computed from the same traces file, excluding the first request.

| stage | median ms (all 12) | median ms (warm 11) | p95 ms (all) | first request ms | share of warm pipeline |
|---|---:|---:|---:|---:|---:|
| `retrieve` | 109 | 104 | 41,586 | 92,059 | 4.8% |
| `rerank` | 302 | 300 | 2,454 | 5,050 | 11.2% |
| `generate` | 2,054 | 1,995 | 3,820 | 4,405 | 83.8% |
| `rag_pipeline` | 2,495 | 2,428 | 47,850 | 101,524 | — |

* **Generation is the cost once warm**: about 84% of warm
  pipeline time; retrieval is about 5%.
* **The first request is 42× the warm median**, and almost all of that is
  `retrieve` loading the embedding model (92 s) — not the LLM. A service
  that loads the encoder at startup instead of on first use removes it from user latency.
* **Tokens**: 3,225 in, 196 out across the run; cost 0.00 USD at
  the default self-hosted price of zero (hosted prices are configuration, not measurement).

**CI gate** (thresholds fixed in `src/eval_ci.py` before this run) — verdict **PASS**, exit 0:

| gate | result | threshold |
|---|---:|---:|
| retrieval (expected doc retrieved) | 1.00 | ≥ 0.8 |
| grounding (answer contains expected fact) | 0.80 | ≥ 0.7 |
| abstention on out-of-scope questions | 1.00 | ≥ 0.5 |
| false abstention on answerable questions | 0.00 | reported |

Twelve questions: the gate proves the mechanism fails a build on a regression; the rates
themselves are too small a sample to rank models or prompts.

## Limitations

- **12 documents, 12 questions.** One case moves grounding by 10 points. The
  corpus is synthetic HR policy written for this project — retrieval quality
  here is not a result and is not offered as one. *Next:* point the same
  instrumentation at a corpus that was not written to be retrievable.
- **The gates have not yet been shown to catch a regression on this tree.**
  The failure path itself is exercised — feeding a report with grounding 0.500
  prints `EVAL CI FAILED: grounding 0.500 < 0.7` and returns exit 1 — but that
  is the arithmetic, not the gate doing its job. *Next:* degrade retrieval
  deliberately, run the real pipeline, and confirm the build goes red on its
  own.
- **Latency is indicative, not a benchmark.** One process on a shared
  desktop, no warm-up discipline, no repetitions, GPU share varying with
  whatever else holds VRAM (Windows 11, 15.9 GB RAM, GTX 1050 Ti 4 GB, with
  Ollama offloading part of the model to it). The per-stage *breakdown* is the
  deliverable; the absolute milliseconds are not a claim about hardware
  capability.
- **Token counts are real; any cost is modelled.** Local inference is
  billed at zero. `cost.py` takes prices from config, defaulting to 0.0/0.0, so
  the same counts can optionally be priced against an illustrative rate —
  that is a projection and is labelled as one on every surface.

## What this project does NOT demonstrate

- **Anything about Langfuse or Qdrant.** The sink is a local JSONL file and
  retrieval is an in-process dense store. There is no evidence here of
  integrating with a hosted tracing backend or a vector database, and nothing
  in this repository claims otherwise.
- **Faithfulness or relevance scores from an LLM judge.** An earlier version of
  this project scored answers that way. Those numbers were never validated
  against human labels and are not reproduced here — scoring is deterministic,
  described in [Eval](#eval).
- **Sampling, retention, or PII redaction** — the three things a real tracing
  deployment has to solve and this one does not.
- **"No GPU" or CPU-only performance.** The machine has an NVIDIA GTX 1050 Ti
  (4 GB) and Ollama offloads part of the model to it; nothing here measures a
  CPU-only path.

## Reproduction

```bash
python run_traced.py
```

Refuses to run against a dirty worktree or outside a git repository unless
`--allow-dirty` is passed — the summary records `source_commit_sha` and
`worktree_clean`, and a result that cannot be tied to a commit is not
reproducible. Writes `results/traces_<sha8>.jsonl` and
`results/summary_<sha8>.json`.

`eval_ci` and the dashboard both report through one `tracer.stage_latencies()`
and one `tracer.median()`, so the console output, the summary and the web page
cannot disagree about what the traces say. They used to: one took
`sorted(v)[n // 2]` and the other a nearest-rank percentile, and the same set
of traces produced two different `generate` medians for identical data.

## Layout

```
src/
├── tracer.py        local JSONL span sink + median/stage_latencies helpers
├── store.py         in-process dense store (e5 query:/passage: prefixes)
├── rag_observed.py  the instrumented pipeline
├── eval_ci.py       the three gates — exit 1 on a breach
├── cost.py          pure cost modelling, no I/O
└── schema.py        RAGResponse, Document
dashboard/app.py     FastAPI view over runs/traces.jsonl
data/
├── corpus.jsonl     12 synthetic HR policy documents
└── eval_set.jsonl   12 cases (10 answerable + 2 not)
run_traced.py        measured-run entry point → results/
docs/results.md      write-up, filled in after the measured run
runs/traces.jsonl    local sink used by `eval_ci` / the dashboard directly
```
