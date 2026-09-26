"""Git provenance and dirty-tree guard for measured runs.

A result is only trustworthy if we know exactly what code produced it. Before a
run starts, check_provenance answers two questions with plain `git` calls: what
commit is checked out, and is the worktree clean.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DirtyWorktreeError(RuntimeError):
    """Raised when a run is refused: no git repo, or a dirty tree without --allow-dirty."""


def _git(*args: str) -> str:
    """Stripped stdout of `git <args>` run at the repo root; raises if git fails."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except FileNotFoundError as exc:
        raise DirtyWorktreeError("git is not installed") from exc
    if result.returncode != 0:
        raise DirtyWorktreeError(
            f"not a git repository (git {' '.join(args)} failed): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def check_provenance(allow_dirty: bool) -> tuple[str, bool]:
    """Return (source_commit_sha, worktree_clean) or raise DirtyWorktreeError.

    Outside a git repository the run is always refused: the result file is named
    after the commit, so there must be one. A dirty tree is refused unless
    `allow_dirty` is set, in which case it is recorded as worktree_clean=False.
    """
    sha = _git("rev-parse", "HEAD")
    clean = _git("status", "--porcelain") == ""
    if not clean and not allow_dirty:
        raise DirtyWorktreeError(
            "worktree has uncommitted changes; commit or stash them, or pass --allow-dirty"
        )
    return sha, clean
