"""Span tracing to a local JSONL file.

This replaces the hosted Langfuse client the project was written against. The
swap is deliberate and it is a downgrade in exactly one respect: there is no
Langfuse UI, and nothing here proves the project integrates with Langfuse.

What it does prove is the part that was actually worth proving. A tracing SDK
is not magic: it is a context variable holding the current span, a stack
discipline for nesting, and a writer. Rebuilding that surface in ~150 lines
makes the trace model explicit instead of hiding it behind a decorator, and it
removes three external dependencies (an account, two API keys, a network) from
a project whose entire claim is that it measures things locally.

The public surface matches the Langfuse one the call sites already use —
`observe`, `langfuse_context`, `add_score`, `flush` — so `rag_observed.py` and
`eval_ci.py` did not have to be rewritten around a different idea.

Spans land in `runs/traces.jsonl`, one JSON object per line, each carrying its
`trace_id`, `span_id`, `parent_id`, timing and whatever the call site attached.
A file is a deliberate choice over SQLite: traces are append-only, they are
read back in bulk, and a text file can be diffed and committed as evidence.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

TRACE_DIR = Path(os.getenv("TRACE_DIR", Path(__file__).resolve().parents[1] / "runs"))
TRACE_PATH = TRACE_DIR / "traces.jsonl"

_current: ContextVar["Span | None"] = ContextVar("current_span", default=None)
_write_lock = threading.Lock()


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str = "span"
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    input: object = None
    output: object = None
    metadata: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    scores: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 2)

    def to_row(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": round(self.started_at, 6),
            "duration_ms": self.duration_ms,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "usage": self.usage,
            "scores": self.scores,
            "error": self.error,
        }


def _write(row: dict) -> None:
    """One line per span, flushed immediately.

    Buffering would lose the trace of the run that crashed, which is the one
    run whose trace matters most.
    """
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _write_lock, TRACE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def observe(name: str | None = None, as_type: str = "span"):
    """Decorator: run the function inside a span, sync or async.

    A span is recorded even when the function raises — a trace that only covers
    successful calls cannot answer the question anyone actually asks it.
    """

    def decorate(fn):
        span_name = name or fn.__name__

        def _open() -> Span:
            parent = _current.get()
            span = Span(
                name=span_name,
                trace_id=parent.trace_id if parent else uuid.uuid4().hex,
                span_id=uuid.uuid4().hex,
                parent_id=parent.span_id if parent else None,
                kind=as_type,
            )
            return span

        def _close(span: Span, token, error: BaseException | None) -> None:
            span.ended_at = time.time()
            if error is not None:
                span.error = f"{type(error).__name__}: {error}"
            _current.reset(token)
            _write(span.to_row())

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                span = _open()
                token = _current.set(span)
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:
                    _close(span, token, exc)
                    raise
                _close(span, token, None)
                return result

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            span = _open()
            token = _current.set(span)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                _close(span, token, exc)
                raise
            _close(span, token, None)
            return result

        return wrapper

    return decorate


class _Context:
    """The `langfuse_context` surface the call sites already use."""

    @staticmethod
    def get_current_trace_id() -> str | None:
        span = _current.get()
        return span.trace_id if span else None

    @staticmethod
    def get_current_span() -> Span | None:
        return _current.get()

    @staticmethod
    def update_current_observation(**fields) -> None:
        span = _current.get()
        if span is None:
            return
        for key, value in fields.items():
            if key in {"input", "output"}:
                setattr(span, key, value)
            elif key == "usage":
                span.usage.update(value or {})
            elif key == "name":
                span.name = value
            else:
                span.metadata[key] = value

    @staticmethod
    def update_current_trace(**fields) -> None:
        """Trace-level fields live on the root span.

        There is no separate trace object here: the root span *is* the trace,
        and anything attached to it is recoverable by grouping on `trace_id`.
        """
        _Context.update_current_observation(**fields)


langfuse_context = _Context()


def add_score(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """Attach an eval score to a trace as its own row.

    Scores arrive after the span they describe has closed — an eval runs on the
    output — so they cannot be written into it. A separate row keyed by
    `trace_id` keeps the trace append-only and joins back at read time.
    """
    _write(
        {
            "trace_id": trace_id,
            "span_id": uuid.uuid4().hex,
            "parent_id": None,
            "name": name,
            "kind": "score",
            "started_at": round(time.time(), 6),
            "duration_ms": None,
            "value": value,
            "comment": comment,
        }
    )


def flush() -> None:
    """A no-op, kept so call sites need no edit.

    Every row is written and flushed as its span closes, so there is nothing
    pending at exit. The hosted client batched over the network; a local file
    has no reason to.
    """


def read_traces(path: Path = TRACE_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
