"""Latency statistics — src/stats.py. Pure functions over span rows."""
import pytest

from src import stats

# ── median ──


def test_median_of_odd_length_list_is_the_middle_value():
    assert stats.median([5.0, 1.0, 3.0]) == 3.0


def test_median_of_even_length_list_averages_the_middle_pair():
    assert stats.median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_median_of_single_value_list_is_that_value():
    assert stats.median([42.0]) == 42.0


def test_median_of_empty_list_is_zero():
    assert stats.median([]) == 0.0


# ── p95 ──


def test_p95_of_a_small_sample_interpolates_between_the_top_two_values():
    # sorted [1, 2, 3, 4, 5], rank = 0.95 * 4 = 3.8 -> between index 3 (4) and 4 (5)
    assert stats.p95([5.0, 1.0, 3.0, 2.0, 4.0]) == pytest.approx(4.8)


def test_p95_of_a_single_value_is_that_value():
    assert stats.p95([42.0]) == 42.0


def test_p95_of_an_empty_list_is_zero():
    assert stats.p95([]) == 0.0


# ── cold start ──


def test_cold_start_ratio_compares_first_request_to_the_median_of_the_rest():
    # first request 90ms, rest [10, 10] -> median 10 -> ratio 9.0
    assert stats.cold_start_ratio([90.0, 10.0, 10.0]) == pytest.approx(9.0)


def test_cold_start_ratio_is_none_with_fewer_than_two_requests():
    assert stats.cold_start_ratio([90.0]) is None
    assert stats.cold_start_ratio([]) is None


# ── per-stage summaries ──


def test_stage_latencies_ignores_score_rows_and_summarises_by_stage():
    rows = [
        {"kind": "span", "name": "retrieve", "duration_ms": 10.0},
        {"kind": "span", "name": "retrieve", "duration_ms": 20.0},
        {"kind": "score", "name": "abstained", "value": 1.0, "duration_ms": None},
    ]
    stages = stats.stage_latencies(rows)

    assert stages["retrieve"]["n"] == 2
    assert stages["retrieve"]["median_ms"] == pytest.approx(15.0)
    assert stages["retrieve"]["min_ms"] == 10.0
    assert stages["retrieve"]["max_ms"] == 20.0
    assert "abstained" not in stages


def test_stage_summary_share_is_n_times_median_over_root_total():
    # generate: n=3, median 10 -> 30 / root total 100 = 30%. A plain sum of
    # generate durations (10 + 10 + 40 = 60) would give 60%; the recorded
    # summaries use n × median, and so must every surface that shows a share.
    rows = [
        {"kind": "span", "name": "rag_pipeline", "duration_ms": 50.0},
        {"kind": "span", "name": "rag_pipeline", "duration_ms": 50.0},
        {"kind": "span", "name": "generate", "duration_ms": 10.0},
        {"kind": "span", "name": "generate", "duration_ms": 10.0},
        {"kind": "span", "name": "generate", "duration_ms": 40.0},
    ]
    stages = stats.stage_summary(rows)

    assert stages["generate"]["share_of_total"] == 30.0
    assert stages["rag_pipeline"]["share_of_total"] is None
    assert stages["generate"]["p95_ms"] == pytest.approx(37.0)
    assert stats.root_durations(rows) == [50.0, 50.0]
