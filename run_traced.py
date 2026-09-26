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


def build_summary(
    spans: list[dict],
    gate_report,
    gate_exit_code: int,
    settings,
    source_commit_sha: str,
    worktree_clean: bool,
) -> dict:
    from src.eval_ci import gate_checks
    from src.stats import cold_start_ratio, root_durations, stage_summary

    stages = stage_summary(spans)
    roots = root_durations(spans)

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
        "trace_count": len(roots),
        "stages": stages,
        "cold_start_ratio": cold_start_ratio(roots),
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
            "total": tokens_in + tokens_out,
        },
        "cost_usd": round(total_cost, 6),
        "gates": {
            **{
                name: {"result": value, "min": floor}
                for name, value, floor in gate_checks(gate_report)
            },
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
    from src.eval_ci import gate_failures, load_cases, score_cases
    from src.tracer import read_traces

    settings = get_settings()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    short_sha = sha[:8]

    before = len(read_traces())
    report = score_cases(load_cases())

    all_rows = read_traces()
    new_rows = all_rows[before:]
    spans = [r for r in new_rows if r.get("kind") != "score"]

    dest = RESULTS_DIR / f"traces_{short_sha}.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for row in new_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    exit_code = 1 if gate_failures(report) else 0

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
