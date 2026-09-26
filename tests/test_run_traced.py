"""Refusal logic for the measured-run entry point — subprocess is always mocked."""
import subprocess

import run_traced
from src import provenance


def _fake_run(returncodes_and_stdout):
    """Return a stand-in for subprocess.run that answers calls in order."""
    calls = iter(returncodes_and_stdout)

    def _inner(cmd, **kwargs):
        returncode, stdout, stderr = next(calls)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    return _inner


def test_main_returns_exit_code_one_and_refuses_on_a_dirty_tree(monkeypatch, capsys):
    monkeypatch.setattr(
        provenance.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, " M src/tracer.py\n", "")]),
    )

    exit_code = run_traced.main([])

    assert exit_code == 1
    assert "refusing to run" in capsys.readouterr().err


def test_main_returns_exit_code_one_outside_a_git_repository(monkeypatch, capsys):
    monkeypatch.setattr(
        provenance.subprocess, "run",
        _fake_run([(128, "", "fatal: not a git repository")]),
    )

    exit_code = run_traced.main([])

    assert exit_code == 1
    assert "refusing to run" in capsys.readouterr().err


def test_main_proceeds_past_the_refusal_check_with_allow_dirty(monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, " M src/tracer.py\n", "")]),
    )
    called = {}

    def fake_run_measured(sha, worktree_clean):
        called["sha"] = sha
        called["worktree_clean"] = worktree_clean
        return 0

    monkeypatch.setattr(run_traced, "run_measured", fake_run_measured)

    exit_code = run_traced.main(["--allow-dirty"])

    assert exit_code == 0
    assert called == {"sha": "abc123def456", "worktree_clean": False}
