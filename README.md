# LLM Observability — traces, a stage breakdown, and a CI gate that can fail

A RAG pipeline that reports only end-to-end latency tells you a request was
slow. This one tells you **which of three stages** was slow, on every request,
and fails the build when quality drops below a threshold fixed before the run.

## Problem

RAG pipelines fail quietly. A hallucination returns HTTP 200. A prompt tweak
breaks three eval cases and nobody notices until a user does. One request burns
5× the expected tokens and the log line says `answered ok`. Ordinary logging
gives you prose; what you need is structured per-request data you can aggregate.

## Structure

### Entry points

Run `python run_traced.py` first — it is the run the published numbers come from.

| Command | Reads | Writes |
|---|---|---|
| `python run_traced.py` | `data/eval_set.jsonl`, `data/corpus.jsonl`, Ollama | appends to `runs/traces.jsonl`; this run's rows → `results/traces_<sha8>.jsonl`; summary → `results/summary_<sha8>.json`; exit 1 if a gate breaks |
| `python -m src.eval_ci` | same as above | appends to `runs/traces.jsonl`; console report; **exit code** is the CI quality gate |
| `uvicorn dashboard.app:app --port 8080` | `runs/traces.jsonl` | nothing — read-only view at http://127.0.0.1:8080 |

`python -m pytest -q` runs the model-free tests (no Ollama, no encoder).

### Request flow

```
question
  → src/rag_observed.py  answer_question()          root span "rag_pipeline"
       → _retrieve()   src/store.py DenseStore.search  span "retrieve"
       → _rerank()     cross-encoder                   span "rerank"
       → _generate()   Ollama /api/chat                span "generate" (token usage)
  → src/tracer.py  observe() closes each span → one JSON line → runs/traces.jsonl
  → src/eval_ci.py score_cases() scores the answer → add_score() rows, same file
  → run_traced.py  copies this run's rows → results/traces_<sha8>.jsonl
                   src/stats.py stage_summary() → results/summary_<sha8>.json
```

### Code map

| File | What it is |
|---|---|
| `run_traced.py` | measured-run entry point: refuses a dirty tree, runs the eval, writes `results/` |
| `src/eval_ci.py` | eval set loader, the one scoring loop (`score_cases`), the three gates, CI exit code |
| `src/rag_observed.py` | the pipeline: `answer_question` and its three traced stages; `Document`, `RAGResponse` |
| `src/tracer.py` | span tracing: `observe`, `current_trace_id`, `annotate`, `add_score`, `read_traces` |
| `src/stats.py` | every latency statistic: median, p95, per-stage summary and share, cold-start ratio |
| `src/store.py` | in-process dense store over the corpus (e5 `query:`/`passage:` prefixes) |
| `src/cost.py` | token counts × configured price → `RequestCost`; pure, no I/O |
| `src/config.py` | settings (Ollama URL/model, encoder, top-k, prices) from env / `.env` |
| `src/__init__.py` | package marker |
| `dashboard/app.py` | FastAPI read-only dashboard over `runs/traces.jsonl` |
| `data/corpus.jsonl` | 12 synthetic HR-policy documents (Arabic) |
| `data/eval_set.jsonl` | 12 eval cases: 10 answerable, 2 not |
| `results/summary_b382cd4e.json` | the published summary, written by `run_traced.py` at `b382cd4e` |
| `results/traces_b382cd4e.jsonl` | every span and score row of that run |
| `README.md` | this file |
| `docs/results.md` | the full results write-up |
| `docs/DESIGN.md` | design decisions and why |
| `tests/test_eval_ci.py` | scoring loop, gate rates and exit codes |
| `tests/test_run_traced.py` | dirty-tree / not-a-repo refusal logic |
| `tests/test_tracer.py` | span nesting, timing, JSONL sink |
| `tests/test_stats.py` | median, p95, share of total, cold start |
| `tests/test_dashboard.py` | dashboard aggregation without a server |
| `tests/test_cost.py` | cost arithmetic |
| `tests/test_response_models.py` | `Document` / `RAGResponse` validation |
| `tests/__init__.py` | package marker |
| `requirements.txt` | full runtime dependencies |
| `requirements-ci.txt` | the subset the model-free tests need |
| `pyproject.toml` | project metadata, ruff config |
| `.env.example` | optional settings, all defaulted |
| `.github/workflows/tests.yml` | CI: runs the tests |
| `.gitignore`, `.gitattributes`, `LICENSE` | repository housekeeping |

`runs/` is created on first run and is not tracked.

### Read the code in this order

1. `src/rag_observed.py` — what a request does and which span each stage writes.
2. `src/tracer.py` — how a span becomes a JSON line.
3. `src/eval_ci.py` — how an answer is scored and gated.
4. `run_traced.py` — how a run becomes a result file tied to a commit.
5. `src/stats.py` — how the numbers in the summary are computed.
6. `dashboard/app.py` — the same numbers, live, over the sink.

## Architecture

```
answer_question(q)
  └─ rag_pipeline ──────────────── trace root
       ├─ retrieve    dense search over 12 docs, intfloat/multilingual-e5-base
       ├─ rerank      cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 → top 3
       └─ generate    gemma3:4b via Ollama, real token counts read back

Every span → runs/traces.jsonl (one JSON object per line).
```

**The sink is a local file, not a hosted service.** `src/tracer.py` is a
context variable holding the current span, a stack discipline for nesting, and
a writer. No account, no keys, no network. Why, and what that gives up:
[docs/DESIGN.md](docs/DESIGN.md).

## Run

```bash
pip install -r requirements.txt
ollama serve && ollama pull gemma3:4b

python run_traced.py                       # the measured run → results/
python -m src.eval_ci                      # exit 0 = every gate held
uvicorn dashboard.app:app --port 8080      # http://127.0.0.1:8080
```

No keys, no network, no vector-DB container.

## Eval

Twelve cases in `data/eval_set.jsonl` — 10 answerable from the 12-document HR
corpus, 2 deliberately not. Three checks, thresholds fixed **before** the run
(`RETRIEVAL_MIN`, `GROUNDING_MIN`, `ABSTENTION_MIN` in `src/eval_ci.py`):

| Check | Threshold |
|---|---|
| retrieval — the document holding the answer came back at all | ≥ 0.80 |
| grounding — the answer contains the fact the eval set requires | ≥ 0.70 |
| abstention — on an unanswerable question, it declined | ≥ 0.50 |

**Scored by a program, not by a judge.** Every check resolves against the
corpus or against a string the eval set fixed in advance, so verdicts replay
exactly from the saved traces. Why not an LLM judge:
[docs/DESIGN.md](docs/DESIGN.md#why-deterministic-scoring-and-not-an-llm-judge).

## Results

Measured at commit `b382cd4e` on a clean tree, `gemma3:4b` through local
Ollama, 12 questions, 48 spans
([`summary_b382cd4e.json`](results/summary_b382cd4e.json),
[`traces_b382cd4e.jsonl`](results/traces_b382cd4e.jsonl)):

| stage | median ms (warm 11) | first request ms | share of warm pipeline |
|---|---:|---:|---:|
| `retrieve` | 104 | 92,059 | 4.8% |
| `rerank` | 300 | 5,050 | 11.2% |
| `generate` | 1,995 | 4,405 | 83.8% |
| `rag_pipeline` | 2,428 | 101,524 | — |

Gate **PASS**, exit 0: retrieval 1.00, grounding 0.80, abstention 1.00, false
abstention 0.00. Generation is the cost once warm; the first request is 42× the
warm median, almost all of it the encoder loading. All-request medians, p95,
tokens and how to read them: [docs/results.md](docs/results.md).

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
- **The dashboard covers the whole sink, a summary covers one run.** Both use
  `src/stats.py`, so the same rows give the same figures, but
  `runs/traces.jsonl` accumulates every run until it is deleted.

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
`results/summary_<sha8>.json` for the commit it ran against; the published
files are the ones for `b382cd4e`. The summary's stages block can be
recomputed from its traces file with `src.stats.stage_summary()`.
