"""Refusal logic for the measured-run entry point — subprocess is always mocked."""
import subprocess

import pytest

import run_traced


def _fake_run(returncodes_and_stdout):
    """Return a stand-in for subprocess.run that answers calls in order."""
    calls = iter(returncodes_and_stdout)

    def _inner(cmd, cwd=None, capture_output=None, text=None):
        returncode, stdout, stderr = next(calls)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    return _inner


def test_check_repo_state_returns_sha_and_clean_true_on_a_clean_tree(monkeypatch):
    monkeypatch.setattr(
        run_traced.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, "", "")]),
    )

    sha, clean = run_traced.check_repo_state(allow_dirty=False)

    assert sha == "abc123def456"
    assert clean is True


def test_check_repo_state_raises_on_a_dirty_tree_without_allow_dirty(monkeypatch):
    monkeypatch.setattr(
        run_traced.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, " M src/tracer.py\n", "")]),
    )

    with pytest.raises(run_traced.DirtyTreeError):
        run_traced.check_repo_state(allow_dirty=False)


def test_check_repo_state_allows_a_dirty_tree_when_allow_dirty_is_passed(monkeypatch):
    monkeypatch.setattr(
        run_traced.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, " M src/tracer.py\n", "")]),
    )

    sha, clean = run_traced.check_repo_state(allow_dirty=True)

    assert sha == "abc123def456"
    assert clean is False


def test_check_repo_state_raises_outside_a_git_repository(monkeypatch):
    monkeypatch.setattr(
        run_traced.subprocess, "run",
        _fake_run([(128, "", "fatal: not a git repository")]),
    )

    with pytest.raises(run_traced.NotAGitRepoError):
        run_traced.check_repo_state(allow_dirty=False)


def test_main_returns_exit_code_one_and_refuses_on_a_dirty_tree(monkeypatch, capsys):
    monkeypatch.setattr(
        run_traced.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, " M src/tracer.py\n", "")]),
    )

    exit_code = run_traced.main([])

    assert exit_code == 1
    assert "refusing to run" in capsys.readouterr().err


def test_main_returns_exit_code_one_outside_a_git_repository(monkeypatch, capsys):
    monkeypatch.setattr(
        run_traced.subprocess, "run",
        _fake_run([(128, "", "fatal: not a git repository")]),
    )

    exit_code = run_traced.main([])

    assert exit_code == 1
    assert "refusing to run" in capsys.readouterr().err


def test_main_proceeds_past_the_refusal_check_with_allow_dirty(monkeypatch):
    monkeypatch.setattr(
        run_traced.subprocess, "run",
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
