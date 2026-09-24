# LLM Observability — Evaluation Results

<!-- RESULTS: filled from results/summary_<sha8>.json after the measured run -->

This file is a placeholder. It is filled in by copying the fields written to
`results/summary_<sha8>.json` after running:

```bash
python run_traced.py
```

`run_traced.py` refuses to run against a dirty worktree or outside a git
repository (unless `--allow-dirty` is passed), so every number that ends up
here is tied to a specific `source_commit_sha` and a recorded `worktree_clean`
state — no figure is written by hand.

## What will be reported

### Environment

Model, Ollama host, hardware string (this machine: Windows 11, 15.9 GB RAM,
NVIDIA GTX 1050 Ti 4 GB, with Ollama offloading part of the model to it), and
the run timestamp.

### Traces

Span and trace counts, and for each stage (`retrieve`, `rerank`, `generate`,
`rag_pipeline`): sample size, median latency, p95 latency, and share of total
wall clock. `eval_ci` and the dashboard both read these through the same
`tracer.median()` / `tracer.stage_latencies()` helpers, so the three surfaces
cannot disagree about what the traces say — see the correction note below.

### Cold start

The ratio of the first request's `rag_pipeline` duration to the median of the
rest. The first request pays to load the encoder, the cross-encoder and the
model; an average would bury that inside a meaningless overall mean.

### Tokens and cost

Total input/output tokens read back from Ollama, and the cost computed from
them. Local inference is billed at zero; `cost.py` takes prices from config
(default 0.0/0.0), so any non-zero figure here is a labelled projection against
an illustrative rate, not a real invoice.

### Gates

The three pre-registered thresholds (retrieval ≥ 0.80, grounding ≥ 0.70,
abstention ≥ 0.50), each check's result, and the overall verdict
(`EVAL CI PASSED` / `EVAL CI FAILED`, with the process exit code).

## Scored by a program, not a judge

This eval does not call an LLM to grade its own output. An earlier version of
this project did — a model scoring another model's output, with the judge
itself never validated against human labels. That produces a number nobody can
defend: when the judge is wrong there is no way to find out, and
"faithfulness 0.82" reads like a measurement while being an opinion with a
decimal point.

All three checks resolve against the corpus or against a string the eval set
fixed in advance, so the verdicts replay exactly from the saved traces.

One detail worth keeping visible: Arabic comparisons strip tashkeel and
tatweel from **both** sides before comparing, because an answer writing `بعد`
and an eval key writing `بُعد` are the same word. Nothing else is folded —
hamza and alef forms distinguish real words, and folding those would start
passing wrong answers.

## Correction on record

An earlier draft of this file reported specific span-latency numbers computed
locally. Those numbers came from an uncommitted tree — no `source_commit_sha`,
no `worktree_clean` check, no `results/summary_<sha8>.json` backing them — so
they cannot be reproduced from a recorded state and are not repeated here.
They also predate the fix that made `eval_ci` and the dashboard share one
`tracer.median()` (previously one took `sorted(v)[n // 2]` and the other a
nearest-rank percentile, so the same traces produced two different `generate`
medians). The next run through `run_traced.py` replaces this section with
numbers that carry their own provenance.

## What this will not establish

* **Anything about Langfuse or Qdrant.** The sink is a local JSONL file and
  retrieval is an in-process dense store; neither hosted service is part of
  this project.
* **Whether the gates catch a real regression.** Passing on a clean run only
  exercises the arithmetic. The next useful step is to degrade retrieval
  deliberately, run the real pipeline, and confirm the build goes red on its
  own.
* **12 documents, 12 questions.** One case moves grounding by 10 points. The
  corpus is synthetic HR policy written for this project; retrieval quality
  here is not a result and is not offered as one.
* **A latency benchmark.** One process on a shared desktop, no warm-up
  discipline, no repetitions, GPU share varying with whatever else holds VRAM.
  The span *breakdown* is the deliverable; the absolute milliseconds are not a
  claim about hardware capability, and this is not a CPU-only machine.
* **Token counts are real; costs are modelled.** Local inference is billed at
  zero, and any hosted figure is a rate multiplied by a count.

## Reproduce

```bash
pip install -r requirements.txt
ollama serve
ollama pull gemma3:4b
python run_traced.py
```

Traces land in `results/traces_<sha8>.jsonl`, one JSON object per span —
greppable, diffable, and readable without a UI. The gate verdict and every
figure above land in `results/summary_<sha8>.json`.
