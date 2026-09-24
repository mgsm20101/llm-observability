# Results

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
medians). The numbers above replace them and carry their own provenance.

## What this does not establish

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
