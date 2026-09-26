"""Eval CI gate on synthetic traces — no model, no Ollama, no retrieval."""
from src import eval_ci


class _FakeSettings:
    price_input_per_mtok = 0.0
    price_output_per_mtok = 0.0


def _case(id_, answerable, retrieved=None, contains=None, abstained=False):
    return eval_ci.CaseResult(
        id=id_,
        answerable=answerable,
        retrieved_expected=retrieved,
        contains_expected=contains,
        abstained=abstained,
        trace_id=f"trace-{id_}",
        cost_usd=0.0,
        tokens=10,
    )


def test_report_rates_are_none_when_no_cases_apply():
    report = eval_ci.Report(cases=[])

    assert report.retrieval is None
    assert report.grounding is None
    assert report.abstention is None
    assert report.false_abstention is None


def test_report_computes_retrieval_and_grounding_over_answerable_cases_only():
    report = eval_ci.Report(cases=[
        _case("E1", True, retrieved=True, contains=True),
        _case("E2", True, retrieved=True, contains=False),
        _case("E3", False, abstained=True),  # unanswerable, excluded from grounding
    ])

    assert report.retrieval == 1.0
    assert report.grounding == 0.5


def test_report_computes_abstention_over_unanswerable_cases_only():
    report = eval_ci.Report(cases=[
        _case("E1", False, abstained=True),
        _case("E2", False, abstained=False),
        _case("E3", True, retrieved=True, contains=True),  # answerable, excluded
    ])

    assert report.abstention == 0.5


def test_report_false_abstention_flags_answerable_cases_that_declined():
    report = eval_ci.Report(cases=[
        _case("E1", True, retrieved=True, contains=True, abstained=False),
        _case("E2", True, retrieved=True, contains=False, abstained=True),
    ])

    assert report.false_abstention == 0.5


def test_gate_passes_and_returns_exit_code_zero_when_all_thresholds_hold(monkeypatch, capsys):
    report = eval_ci.Report(cases=[
        _case(f"E{i}", True, retrieved=True, contains=True) for i in range(10)
    ] + [
        _case("U1", False, abstained=True),
        _case("U2", False, abstained=True),
    ])
    monkeypatch.setattr(eval_ci, "read_traces", lambda: [])

    exit_code = eval_ci.print_and_gate(report, _FakeSettings())

    assert exit_code == 0
    assert "EVAL CI PASSED" in capsys.readouterr().out


def test_gate_fails_and_returns_exit_code_one_when_grounding_drops_below_threshold(
    monkeypatch, capsys
):
    # 2/10 contain the expected fact → grounding 0.200 < 0.70
    cases = [
        _case(f"E{i}", True, retrieved=True, contains=(i < 2)) for i in range(10)
    ]
    report = eval_ci.Report(cases=cases)
    monkeypatch.setattr(eval_ci, "read_traces", lambda: [])

    exit_code = eval_ci.print_and_gate(report, _FakeSettings())
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "EVAL CI FAILED" in out
    assert "grounding 0.200 < 0.7" in out


def test_gate_reports_stage_latencies_from_the_trace_sink(monkeypatch, capsys):
    report = eval_ci.Report(cases=[_case("E1", True, retrieved=True, contains=True)])
    synthetic_rows = [
        {"kind": "span", "name": "retrieve", "duration_ms": 12.0},
        {"kind": "span", "name": "generate", "duration_ms": 88.0},
    ]
    monkeypatch.setattr(eval_ci, "read_traces", lambda: synthetic_rows)

    eval_ci.print_and_gate(report, _FakeSettings())
    out = capsys.readouterr().out

    assert "retrieve" in out
    assert "generate" in out


def test_fold_treats_diacritic_variants_of_a_word_as_equal():
    assert eval_ci.fold("بُعد") == eval_ci.fold("بعد")


class _Doc:
    def __init__(self, source):
        self.source = source


class _Result:
    def __init__(self, answer, sources, trace_id):
        self.answer = answer
        self.sources = [_Doc(s) for s in sources]
        self.trace_id = trace_id
        self.cost = None


def test_score_cases_scores_each_case_and_writes_scores_in_a_fixed_order(monkeypatch):
    cases = [
        {"id": "E1", "question": "q1", "answerable": True,
         "expected_doc": "leave", "answer_contains": ["21"]},
        {"id": "U1", "question": "q2", "answerable": False},
    ]
    answers = {
        "q1": _Result("21 يوما", ["leave"], "t1"),
        "q2": _Result("لا تتوفر معلومات كافية", [], "t2"),
    }
    written = []
    monkeypatch.setattr(eval_ci, "add_score",
                        lambda trace_id, name, value: written.append((trace_id, name, value)))

    report = eval_ci.score_cases(cases, answer=answers.__getitem__)

    assert written == [
        ("t1", "retrieved_expected", 1.0),
        ("t1", "contains_expected", 1.0),
        ("t1", "abstained", 0.0),
        ("t2", "abstained", 1.0),
    ]
    assert report.retrieval == 1.0
    assert report.grounding == 1.0
    assert report.abstention == 1.0
    assert eval_ci.gate_failures(report) == []
