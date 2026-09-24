"""Tracer: span context/nesting, timing, the local JSONL sink, and shared median()."""
import json

import pytest

from src import tracer


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """Redirect the sink to a scratch file so tests never touch runs/traces.jsonl."""
    trace_path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(tracer, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(tracer, "TRACE_PATH", trace_path)
    return trace_path


# ── timing, driven by explicit (fake) timestamps rather than the wall clock ──


def test_span_duration_ms_is_computed_from_start_and_end_times():
    span = tracer.Span(
        name="generate", trace_id="t1", span_id="s1", parent_id=None,
        started_at=100.0,
    )
    span.ended_at = 100.25

    assert span.duration_ms == pytest.approx(250.0)


def test_span_duration_ms_is_none_while_the_span_is_still_open():
    span = tracer.Span(name="generate", trace_id="t1", span_id="s1", parent_id=None)

    assert span.duration_ms is None


def test_span_to_row_carries_fixed_timestamps_through_unchanged():
    span = tracer.Span(
        name="retrieve", trace_id="t1", span_id="s1", parent_id="root",
        started_at=10.0,
    )
    span.ended_at = 10.5

    row = span.to_row()

    assert row["started_at"] == 10.0
    assert row["duration_ms"] == pytest.approx(500.0)
    assert row["parent_id"] == "root"


# ── context / nesting via the decorator ──


def test_nested_spans_share_trace_id_and_link_parent(sink):
    @tracer.observe(name="child")
    def child():
        return 1

    @tracer.observe(name="parent")
    def parent():
        return child()

    parent()
    rows = {r["name"]: r for r in tracer.read_traces(sink)}

    assert rows["parent"]["parent_id"] is None
    assert rows["child"]["parent_id"] == rows["parent"]["span_id"]
    assert rows["child"]["trace_id"] == rows["parent"]["trace_id"]
    assert rows["child"]["duration_ms"] is not None


def test_sibling_spans_under_the_same_trace_do_not_share_a_parent(sink):
    @tracer.observe(name="stage")
    def stage():
        return 1

    @tracer.observe(name="root")
    def root():
        stage()
        stage()

    root()
    rows = tracer.read_traces(sink)
    stages = [r for r in rows if r["name"] == "stage"]

    assert len(stages) == 2
    assert stages[0]["span_id"] != stages[1]["span_id"]
    assert {r["trace_id"] for r in rows} == {rows[0]["trace_id"]}


def test_span_is_recorded_even_when_the_function_raises(sink):
    @tracer.observe(name="failing")
    def boom():
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        boom()

    rows = tracer.read_traces(sink)
    assert len(rows) == 1
    assert rows[0]["error"] == "ValueError: bad input"
    assert rows[0]["duration_ms"] is not None


def test_current_span_is_cleared_after_a_span_closes(sink):
    @tracer.observe(name="stage")
    def work():
        assert tracer.langfuse_context.get_current_span() is not None

    work()
    assert tracer.langfuse_context.get_current_span() is None


def test_update_current_observation_sets_input_output_and_usage(sink):
    @tracer.observe(name="generate", as_type="generation")
    def generate():
        tracer.langfuse_context.update_current_observation(
            input="q", output="a", usage={"input": 10, "output": 5}
        )

    generate()
    row = tracer.read_traces(sink)[0]

    assert row["input"] == "q"
    assert row["output"] == "a"
    assert row["usage"] == {"input": 10, "output": 5}
    assert row["kind"] == "generation"


# ── the shared median() ──


def test_median_of_odd_length_list_is_the_middle_value():
    assert tracer.median([5.0, 1.0, 3.0]) == 3.0


def test_median_of_even_length_list_averages_the_middle_pair():
    assert tracer.median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_median_of_single_value_list_is_that_value():
    assert tracer.median([42.0]) == 42.0


def test_median_of_empty_list_is_zero():
    assert tracer.median([]) == 0.0


# ── the JSONL sink: round trip and aggregation ──


def test_jsonl_store_round_trips_written_rows(sink):
    tracer._write({"trace_id": "t1", "span_id": "s1", "name": "retrieve", "duration_ms": 12.5})
    tracer._write({"trace_id": "t1", "span_id": "s2", "name": "generate", "duration_ms": 40.0})

    rows = tracer.read_traces(sink)

    assert [r["name"] for r in rows] == ["retrieve", "generate"]
    assert sink.exists()
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # one JSON object per line, as documented
    assert json.loads(lines[0])["span_id"] == "s1"


def test_read_traces_returns_empty_list_when_the_sink_does_not_exist(tmp_path):
    missing = tmp_path / "nope.jsonl"
    assert tracer.read_traces(missing) == []


def test_add_score_writes_a_score_row_keyed_by_trace_id(sink):
    tracer.add_score("t1", "grounded", 1.0, comment="ok")

    rows = tracer.read_traces(sink)
    assert rows[0]["kind"] == "score"
    assert rows[0]["trace_id"] == "t1"
    assert rows[0]["value"] == 1.0


def test_stage_latencies_ignores_score_rows_and_summarises_by_stage(sink):
    rows = [
        {"kind": "span", "name": "retrieve", "duration_ms": 10.0},
        {"kind": "span", "name": "retrieve", "duration_ms": 20.0},
        {"kind": "score", "name": "abstained", "value": 1.0, "duration_ms": None},
    ]
    stages = tracer.stage_latencies(rows)

    assert stages["retrieve"]["n"] == 2
    assert stages["retrieve"]["median_ms"] == pytest.approx(15.0)
    assert stages["retrieve"]["min_ms"] == 10.0
    assert stages["retrieve"]["max_ms"] == 20.0
    assert "abstained" not in stages
