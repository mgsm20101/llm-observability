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

Latency is read back from the trace rows rather than re-timed here.

`score_cases()` is the one eval loop in the project: this module's `run()` and
`run_traced.run_measured()` both call it, and `gate_failures()` is the one
place the thresholds are applied.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings
from .rag_observed import answer_question
from .stats import stage_latencies
from .tracer import TRACE_PATH, add_score, flush, read_traces

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval_set.jsonl"

# Pre-registered, before the run. Changing one of these after seeing a result
# is how a gate stops being a gate.
RETRIEVAL_MIN = 0.80
GROUNDING_MIN = 0.70
ABSTENTION_MIN = 0.50

ABSTENTION_MARKERS = ("لا تتوفر", "لا توجد", "غير متوفر", "لا أعرف", "لم أجد")


def fold(text: str) -> str:
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


def load_cases(path: Path = EVAL_SET_PATH) -> list[dict]:
    """The eval set, one JSON object per line."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_case(case: dict, result) -> CaseResult:
    """Score one pipeline result against its eval case. Pure — no I/O."""
    folded = fold(result.answer)
    abstained = any(fold(m) in folded for m in ABSTENTION_MARKERS)
    answerable = bool(case["answerable"])

    retrieved = contains = None
    if answerable:
        retrieved = any(d.source == case["expected_doc"] for d in result.sources)
        wanted = case.get("answer_contains") or []
        contains = all(fold(w) in folded for w in wanted)

    return CaseResult(
        id=case["id"],
        answerable=answerable,
        retrieved_expected=retrieved,
        contains_expected=contains,
        abstained=abstained,
        trace_id=result.trace_id or "unknown",
        cost_usd=result.cost.cost_usd if result.cost else 0.0,
        tokens=result.cost.total_tokens if result.cost else 0,
    )


def score_cases(
    cases: list[dict],
    answer: Callable[[str], object] = answer_question,
    verbose: bool = False,
) -> Report:
    """Run every case through the pipeline, score it, and attach the scores.

    The one eval loop: `run()` below and `run_traced.run_measured` both call
    it. Scores go onto the trace as their own rows, so a failure is
    inspectable from the trace alone rather than only from console output.
    """
    report = Report()
    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"  [{i}/{len(cases)}] {case['question'][:58]}…", flush=True)
        scored = score_case(case, answer(case["question"]))
        report.cases.append(scored)

        if scored.answerable:
            add_score(scored.trace_id, "retrieved_expected",
                      float(bool(scored.retrieved_expected)))
            add_score(scored.trace_id, "contains_expected",
                      float(bool(scored.contains_expected)))
        add_score(scored.trace_id, "abstained", float(scored.abstained))

        if verbose:
            ok = scored.contains_expected if scored.answerable else scored.abstained
            print(f"       {'ok ' if ok else 'MISS'}  tokens={scored.tokens}",
                  flush=True)
    flush()
    return report


def gate_checks(report: Report) -> list[tuple[str, float | None, float]]:
    """(gate, measured rate, pre-registered floor) for each of the three gates."""
    return [
        ("retrieval", report.retrieval, RETRIEVAL_MIN),
        ("grounding", report.grounding, GROUNDING_MIN),
        ("abstention", report.abstention, ABSTENTION_MIN),
    ]


def gate_failures(report: Report) -> list[str]:
    """One line per breached gate; empty means the gate passed.

    A rate of None (no case the gate applies to) is not a failure.
    """
    return [
        f"{name} {value:.3f} < {floor}"
        for name, value, floor in gate_checks(report)
        if value is not None and value < floor
    ]


def run() -> int:
    report = score_cases(load_cases(), verbose=True)
    return print_and_gate(report, get_settings())


def print_and_gate(report: Report, settings) -> int:
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
    labels = {
        "retrieval": "retrieval   (answerable)",
        "grounding": "grounding   (answerable)",
        "abstention": "abstention  (unanswerable)",
    }
    for name, value, floor in gate_checks(report):
        shown = "—" if value is None else f"{value:.3f}"
        verdict = "n/a" if value is None else ("PASS" if value >= floor else "FAIL")
        print(f"  {labels[name]:<28} {shown:>6}   (min {floor})   {verdict}")

    fa = report.false_abstention
    print(f"  {'false abstention':<28} {'—' if fa is None else f'{fa:.3f}':>6}"
          f"   (read the abstention rate only next to this)")

    print("=" * 58)
    failed = gate_failures(report)
    if failed:
        print(f"\nEVAL CI FAILED: {', '.join(failed)}")
        return 1
    print("\nEVAL CI PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
