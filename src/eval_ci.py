"""Eval CI runner — run after every commit that touches prompts or retrieval.

    python -m src.eval_ci

Exit 0 = every pre-registered threshold held. Exit 1 = at least one broke, and
the build should fail.

**Scored by a program, not by a judge.** An earlier version of this gate
scored answers with a second LLM call judging faithfulness and relevance — a
model scoring another model's output, with no validation of the judge itself
against human labels. That produces a number nobody can defend: if the judge
is wrong you cannot tell, and "faithfulness 0.82" reads like a measurement
when it is an opinion with a decimal point. Every check below resolves against
the corpus or against a string the eval set fixed in advance, so the verdicts
replay exactly from the saved traces.

The three things measured here are the three an observability project should be
able to answer about itself:

  retrieval   did the document that holds the answer come back at all?
  grounding   did the answer contain the fact the eval set requires?
  abstention  on a question the corpus cannot answer, did it decline?

Cost and latency come from the traces rather than being recomputed, so the
numbers in this report and the numbers in `runs/traces.jsonl` cannot disagree.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings
from .rag_observed import answer_question
from .tracer import TRACE_PATH, add_score, flush, read_traces, stage_latencies

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval_set.jsonl"

# Pre-registered, before the run. Changing one of these after seeing a result
# is how a gate stops being a gate.
RETRIEVAL_MIN = 0.80
GROUNDING_MIN = 0.70
ABSTENTION_MIN = 0.50

ABSTENTION_MARKERS = ("لا تتوفر", "لا توجد", "غير متوفر", "لا أعرف", "لم أجد")


def _fold(text: str) -> str:
    """Strip Arabic diacritics and tatweel before comparing.

    An answer that says `بعد` and an eval key that says `بُعد` are the same
    word; a raw substring check calls them different and fails a correct
    answer. Only optional orthography is removed — hamza and alef forms are
    left alone, because those distinguish real words and folding them would
    start passing wrong answers.
    """
    out = []
    for ch in unicodedata.normalize("NFC", text):
        if ("ً" <= ch <= "ْ") or ch in "ٰـ":
            continue
        out.append(ch)
    return "".join(out).casefold()


@dataclass
class CaseResult:
    id: str
    answerable: bool
    retrieved_expected: bool | None
    contains_expected: bool | None
    abstained: bool
    trace_id: str
    cost_usd: float
    tokens: int


@dataclass
class Report:
    cases: list[CaseResult] = field(default_factory=list)

    def _rate(self, pick, over) -> float | None:
        rows = [c for c in self.cases if over(c)]
        return round(sum(1 for c in rows if pick(c)) / len(rows), 3) if rows else None

    @property
    def retrieval(self) -> float | None:
        return self._rate(lambda c: c.retrieved_expected, lambda c: c.answerable)

    @property
    def grounding(self) -> float | None:
        return self._rate(lambda c: c.contains_expected, lambda c: c.answerable)

    @property
    def abstention(self) -> float | None:
        return self._rate(lambda c: c.abstained, lambda c: not c.answerable)

    @property
    def false_abstention(self) -> float | None:
        return self._rate(lambda c: c.abstained, lambda c: c.answerable)


def run() -> int:
    settings = get_settings()
    cases = [
        json.loads(line)
        for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = Report()

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['question'][:58]}…", flush=True)
        result = answer_question(case["question"])
        trace_id = result.trace_id or "unknown"
        folded = _fold(result.answer)

        abstained = any(_fold(m) in folded for m in ABSTENTION_MARKERS)
        answerable = bool(case["answerable"])

        retrieved = contains = None
        if answerable:
            retrieved = any(d.source == case["expected_doc"] for d in result.sources)
            wanted = case.get("answer_contains") or []
            contains = all(_fold(w) in folded for w in wanted)

        report.cases.append(
            CaseResult(
                id=case["id"],
                answerable=answerable,
                retrieved_expected=retrieved,
                contains_expected=contains,
                abstained=abstained,
                trace_id=trace_id,
                cost_usd=result.cost.cost_usd if result.cost else 0.0,
                tokens=result.cost.total_tokens if result.cost else 0,
            )
        )

        # Scores go onto the trace, so a failure is inspectable from the trace
        # alone rather than only from this console output.
        if answerable:
            add_score(trace_id, "retrieved_expected", float(bool(retrieved)))
            add_score(trace_id, "contains_expected", float(bool(contains)))
        add_score(trace_id, "abstained", float(abstained))

        mark = "ok " if (contains if answerable else abstained) else "MISS"
        print(f"       {mark}  tokens={report.cases[-1].tokens}", flush=True)

    flush()
    return _print_and_gate(report, settings)


def _print_and_gate(report: Report, settings) -> int:
    rows = read_traces()
    spans = [r for r in rows if r.get("kind") != "score"]
    stages = stage_latencies(rows)

    print("\n" + "=" * 58)
    print("  traces")
    print("=" * 58)
    print(f"  file                : {TRACE_PATH}")
    print(f"  spans recorded      : {len(spans)}")
    for stage, s in stages.items():
        print(f"  {stage:<20}: n={s['n']:<3} median {s['median_ms']:>9.1f} ms"
              f"   (min {s['min_ms']:.1f} / max {s['max_ms']:.1f})")
    print("  max is the first request of the run, which pays to load the")
    print("  encoder, the reranker and the model. Read the median.")

    tokens = sum(c.tokens for c in report.cases)
    print(f"\n  total tokens        : {tokens}")
    print(f"  cost at local rates : ${sum(c.cost_usd for c in report.cases):.6f}"
          f"  (${settings.price_input_per_mtok}/Mtok in,"
          f" ${settings.price_output_per_mtok}/Mtok out)")

    print("\n" + "=" * 58)
    print("  gates — thresholds fixed before the run")
    print("=" * 58)
    checks = [
        ("retrieval   (answerable)", report.retrieval, RETRIEVAL_MIN),
        ("grounding   (answerable)", report.grounding, GROUNDING_MIN),
        ("abstention  (unanswerable)", report.abstention, ABSTENTION_MIN),
    ]
    failed = []
    for label, value, floor in checks:
        shown = "—" if value is None else f"{value:.3f}"
        verdict = "n/a"
        if value is not None:
            ok = value >= floor
            verdict = "PASS" if ok else "FAIL"
            if not ok:
                failed.append(f"{label.split()[0]} {value:.3f} < {floor}")
        print(f"  {label:<28} {shown:>6}   (min {floor})   {verdict}")

    fa = report.false_abstention
    print(f"  {'false abstention':<28} {'—' if fa is None else f'{fa:.3f}':>6}"
          f"   (read the abstention rate only next to this)")

    print("=" * 58)
    if failed:
        print(f"\nEVAL CI FAILED: {', '.join(failed)}")
        return 1
    print("\nEVAL CI PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
