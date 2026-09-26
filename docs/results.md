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

Twelve questions: the gate is a mechanism for failing a build, and the rates
themselves are too small a sample to rank models or prompts.

## Scoring detail

Scoring is deterministic — why is in
[DESIGN.md](DESIGN.md#why-deterministic-scoring-and-not-an-llm-judge).

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
`median()`, now in `src/stats.py` (previously one took `sorted(v)[n // 2]`
and the other a nearest-rank percentile, so the same traces produced two different `generate`
medians). The numbers above replace them and carry their own provenance.

## Limits and reproduction

What these numbers do not establish is listed once, in the README's
[Limitations](../README.md#limitations) and
[What this project does NOT demonstrate](../README.md#what-this-project-does-not-demonstrate);
how to reproduce them is in [Reproduction](../README.md#reproduction).
