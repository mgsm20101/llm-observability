"""Git provenance guard in src/provenance.py — subprocess is always mocked."""
import subprocess

import pytest

from src import provenance


def _fake_run(returncodes_and_stdout):
    """Return a stand-in for subprocess.run that answers calls in order."""
    calls = iter(returncodes_and_stdout)

    def _inner(cmd, **kwargs):
        returncode, stdout, stderr = next(calls)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    return _inner


def test_check_provenance_returns_sha_and_clean_true_on_a_clean_tree(monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, "", "")]),
    )

    sha, clean = provenance.check_provenance(allow_dirty=False)

    assert sha == "abc123def456"
    assert clean is True


def test_check_provenance_raises_on_a_dirty_tree_without_allow_dirty(monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, " M src/tracer.py\n", "")]),
    )

    with pytest.raises(provenance.DirtyWorktreeError):
        provenance.check_provenance(allow_dirty=False)


def test_check_provenance_allows_a_dirty_tree_when_allow_dirty_is_passed(monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess, "run",
        _fake_run([(0, "abc123def456\n", ""), (0, " M src/tracer.py\n", "")]),
    )

    sha, clean = provenance.check_provenance(allow_dirty=True)

    assert sha == "abc123def456"
    assert clean is False


def test_check_provenance_raises_outside_a_git_repository(monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess, "run",
        _fake_run([(128, "", "fatal: not a git repository")]),
    )

    with pytest.raises(provenance.DirtyWorktreeError):
        provenance.check_provenance(allow_dirty=False)
