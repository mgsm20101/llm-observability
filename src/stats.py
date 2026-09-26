"""Every latency statistic in the project, in one place.

`eval_ci` (console report), `run_traced` (the summary JSON) and the dashboard
all compute from these functions. They used not to: one component took
`sorted[n // 2]` and another a nearest-rank percentile, so the same 12 traces
produced `generate` medians of 9763.3 ms and 9559.2 ms. Two components of one
project reporting different medians for identical data is a bug that looks
like a measurement.

Pure functions over span rows (the dicts `tracer.read_traces()` returns) — no
I/O.
"""

from __future__ import annotations

ROOT_SPAN = "rag_pipeline"


def median(values: list[float]) -> float:
    """The real median — the mean of the two middle values when n is even."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def p95(values: list[float]) -> float:
    """Linear-interpolation p95 — the tail figure, not the median."""
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


def _durations(rows: list[dict]) -> dict[str, list[float]]:
    """Span durations grouped by span name; score rows and open spans skipped."""
    by_stage: dict[str, list[float]] = {}
    for row in rows:
        if row.get("kind") == "score" or row.get("duration_ms") is None:
            continue
        by_stage.setdefault(row["name"], []).append(float(row["duration_ms"]))
    return by_stage


def stage_latencies(rows: list[dict]) -> dict[str, dict]:
    """Per-stage timing summary: `{stage: {n, median_ms, min_ms, max_ms}}`.

    The min/max matter here more than they usually would: the first request of
    a run pays for loading the encoder, the reranker and the model, so its span
    durations are an order of magnitude above the rest and would drag any mean.
    """
    return {
        name: {
            "n": len(values),
            "median_ms": round(median(values), 1),
            "min_ms": round(min(values), 1),
            "max_ms": round(max(values), 1),
        }
        for name, values in sorted(_durations(rows).items())
    }


def stage_summary(rows: list[dict]) -> dict[str, dict]:
    """`stage_latencies` plus `p95_ms` and `share_of_total` for each stage.

    `share_of_total` is `n × median / sum(root span durations)`, in percent,
    and None for the root span itself. It is the figure recorded in
    `results/summary_<sha8>.json`, so the formula must not change without
    re-measuring.
    """
    durations = _durations(rows)
    stages = stage_latencies(rows)
    for name, stage in stages.items():
        stage["p95_ms"] = round(p95(durations[name]), 1)

    root_total = sum(durations.get(ROOT_SPAN, [])) or None
    for name, stage in stages.items():
        stage["share_of_total"] = (
            None
            if name == ROOT_SPAN or root_total is None
            else round(100 * stage["n"] * stage["median_ms"] / root_total, 1)
        )
    return stages


def root_durations(rows: list[dict]) -> list[float]:
    """Durations of the root span, one per request, in the order written."""
    return _durations(rows).get(ROOT_SPAN, [])


def cold_start_ratio(root_durations_ms: list[float]) -> float | None:
    """First request's duration over the median of every other request.

    The first request of a run pays to load the encoder, the cross-encoder
    and the model; this ratio is how large that one-time cost is relative to
    a warm request, without needing a warm-up phase to hide it.
    """
    if len(root_durations_ms) < 2:
        return None
    first, *rest = root_durations_ms
    rest_median = median(rest)
    if not rest_median:
        return None
    return round(first / rest_median, 2)
