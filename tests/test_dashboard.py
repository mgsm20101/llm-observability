"""Dashboard aggregation — dashboard/app.py, without starting a server."""
import asyncio

from dashboard import app as dashboard_app


def _run(coro):
    return asyncio.run(coro)


def test_get_metrics_reports_no_traces_message_on_an_empty_sink(monkeypatch):
    monkeypatch.setattr(dashboard_app, "read_traces", lambda: [])

    result = _run(dashboard_app.get_metrics())

    assert "message" in result
    assert "No traces yet" in result["message"]


def test_get_metrics_aggregates_span_and_score_rows(monkeypatch):
    synthetic_rows = [
        {"kind": "span", "name": "rag_pipeline", "duration_ms": 100.0,
         "usage": {}},
        {"kind": "span", "name": "retrieve", "duration_ms": 20.0,
         "usage": {"input": 5, "output": 0}},
        {"kind": "span", "name": "generate", "duration_ms": 70.0,
         "usage": {"input": 30, "output": 15}},
        {"kind": "score", "name": "grounded", "value": 1.0},
        {"kind": "score", "name": "grounded", "value": 0.0},
    ]
    monkeypatch.setattr(dashboard_app, "read_traces", lambda: synthetic_rows)

    result = _run(dashboard_app.get_metrics())

    assert result["span_count"] == 3
    assert result["trace_count"] == 1
    assert result["tokens_in"] == 35
    assert result["tokens_out"] == 15
    assert result["total_tokens"] == 50
    assert result["scores"]["grounded"]["n"] == 2
    assert result["scores"]["grounded"]["mean"] == 0.5


def test_get_metrics_computes_share_of_total_for_non_root_stages(monkeypatch):
    synthetic_rows = [
        {"kind": "span", "name": "rag_pipeline", "duration_ms": 100.0, "usage": {}},
        {"kind": "span", "name": "generate", "duration_ms": 65.0, "usage": {}},
    ]
    monkeypatch.setattr(dashboard_app, "read_traces", lambda: synthetic_rows)

    result = _run(dashboard_app.get_metrics())

    assert result["stages"]["generate"]["share_of_total"] == 65.0
    assert result["stages"]["rag_pipeline"]["share_of_total"] is None


def test_dashboard_ui_route_returns_html_without_hitting_the_sink():
    html = _run(dashboard_app.dashboard_ui())

    assert "<html" in html
    assert "LLM Observability" in html
