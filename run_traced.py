"""Measured-run entry point.

    python run_traced.py [--allow-dirty]

Runs the eval set through the traced pipeline, writes every span to
`results/traces_<sha8>.jsonl`, gates the run with `src.eval_ci`, and writes
`results/summary_<sha8>.json` — span/trace counts, per-stage median and p95
latency, each stage's share of total wall clock, the cold-start ratio (first
request vs. the median of the rest), token counts, computed cost, the gate
thresholds and verdict, model, Ollama host, hardware, timestamp, the commit
the run was taken against, and whether the worktree was clean at run time.

Refuses to run against a dirty worktree, or outside a git repository, unless
`--allow-dirty` is passed: a result that cannot be tied to a commit is not
reproducible, and this project's whole point is that a number here should be
traceable back to the code that produced it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

HARDWARE = "Windows 11 · 15.9 GB RAM · NVIDIA GTX 1050 Ti 4 GB (Ollama GPU-offload)"


class DirtyTreeError(RuntimeError):
    """The worktree has uncommitted changes and --allow-dirty was not passed."""


class NotAGitRepoError(RuntimeError):
    """ROOT is not inside a git repository (or git is unavailable)."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise NotAGitRepoError(result.stderr.strip() or "not a git repository")
    return result.stdout.strip()


def check_repo_state(allow_dirty: bool) -> tuple[str, bool]:
    """Return (commit_sha, worktree_clean); raise if dirty and not allowed.

    Two calls, not one: `rev-parse HEAD` needs at least one commit to exist,
    `status --porcelain` needs nothing but a working tree. Failing either one
    means the run cannot be tied to a commit, which is the refusal this
    function exists to enforce.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    clean = status == ""
    if not clean and not allow_dirty:
        raise DirtyTreeError(
            "worktree has uncommitted changes; commit or stash them, "
            "or pass --allow-dirty"
        )
    return sha, clean


def _p95(values: list[float]) -> float:
    """Linear-interpolation p95 — the tail figure, not the median.

    `tracer.median()` is the shared source of truth for the middle of the
    distribution; p95 is computed here because the CI gate does not need it
    and the shared helper deliberately stays small.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = 0.95 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _cold_start_ratio(root_durations_ms: list[float]) -> float | None:
    """First request's duration over the median of every other request.

    The first request of a run pays to load the encoder, the cross-encoder
    and the model; this ratio is how large that one-time cost is relative to
    a warm request, without needing a warm-up phase to hide it.
    """
    from src.tracer import median

    if len(root_durations_ms) < 2:
        return None
    first, *rest = root_durations_ms
    rest_median = median(rest)
    if not rest_median:
        return None
    return round(first / rest_median, 2)


def build_summary(
    spans: list[dict],
    gate_report,
    gate_exit_code: int,
    settings,
    source_commit_sha: str,
    worktree_clean: bool,
) -> dict:
    from src.tracer import median, stage_latencies

    stages = stage_latencies(spans)
    for name, stage in stages.items():
        values = [
            float(s["duration_ms"])
            for s in spans
            if s.get("name") == name and s.get("duration_ms") is not None
        ]
        stage["p95_ms"] = round(_p95(values), 1)

    root_total = sum(
        float(s["duration_ms"])
        for s in spans
        if s.get("name") == "rag_pipeline" and s.get("duration_ms") is not None
    ) or None
    for name, stage in stages.items():
        stage["share_of_total"] = (
            None
            if name == "rag_pipeline" or root_total is None
            else round(100 * stage["n"] * stage["median_ms"] / root_total, 1)
        )

    root_durations = [
        float(s["duration_ms"])
        for s in spans
        if s.get("name") == "rag_pipeline" and s.get("duration_ms") is not None
    ]

    tokens_in = sum(int((s.get("usage") or {}).get("input") or 0) for s in spans)
    tokens_out = sum(int((s.get("usage") or {}).get("output") or 0) for s in spans)
    total_cost = sum(c.cost_usd for c in gate_report.cases)

    return {
        "model": settings.ollama_model,
        "ollama_host": settings.ollama_base_url,
        "hardware": HARDWARE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_commit_sha": source_commit_sha,
        "worktree_clean": worktree_clean,
        "span_count": len(spans),
        "trace_count": len(root_durations),
        "stages": stages,
        "cold_start_ratio": _cold_start_ratio(root_durations),
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
            "total": tokens_in + tokens_out,
        },
        "cost_usd": round(total_cost, 6),
        "gates": {
            "retrieval": {"result": gate_report.retrieval, "min": 0.80},
            "grounding": {"result": gate_report.grounding, "min": 0.70},
            "abstention": {"result": gate_report.abstention, "min": 0.50},
            "false_abstention": {"result": gate_report.false_abstention},
        },
        "verdict": "PASS" if gate_exit_code == 0 else "FAIL",
        "exit_code": gate_exit_code,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run against a dirty worktree instead of refusing",
    )
    return parser.parse_args(argv)


def run_measured(sha: str, worktree_clean: bool) -> int:
    """The actual traced run: eval set → pipeline → gate → summary files.

    Imports are local to this function so that importing this module (for
    tests of the refusal logic above) never touches Ollama, the encoder or
    the reranker.
    """
    from src.config import get_settings
    from src.eval_ci import EVAL_SET_PATH, ABSTENTION_MARKERS, Report, _fold, CaseResult
    from src.rag_observed import answer_question
    from src.tracer import add_score, flush, read_traces

    settings = get_settings()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    short_sha = sha[:8]

    before = len(read_traces())
    cases = [
        json.loads(line)
        for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = Report()
    for case in cases:
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
        if answerable:
            add_score(trace_id, "retrieved_expected", float(bool(retrieved)))
            add_score(trace_id, "contains_expected", float(bool(contains)))
        add_score(trace_id, "abstained", float(abstained))
    flush()

    all_rows = read_traces()
    new_rows = all_rows[before:]
    spans = [r for r in new_rows if r.get("kind") != "score"]

    dest = RESULTS_DIR / f"traces_{short_sha}.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for row in new_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    checks = [
        ("retrieval", report.retrieval, 0.80),
        ("grounding", report.grounding, 0.70),
        ("abstention", report.abstention, 0.50),
    ]
    exit_code = 0
    for _, value, floor in checks:
        if value is not None and value < floor:
            exit_code = 1

    summary = build_summary(spans, report, exit_code, settings, sha, worktree_clean)
    summary_path = RESULTS_DIR / f"summary_{short_sha}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {dest}")
    print(f"wrote {summary_path}")
    print(f"verdict: {summary['verdict']} (exit {exit_code})")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sha, worktree_clean = check_repo_state(args.allow_dirty)
    except (DirtyTreeError, NotAGitRepoError) as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 1
    return run_measured(sha, worktree_clean)


if __name__ == "__main__":
    raise SystemExit(main())
