"""Span tracing to a local JSONL file.

A tracing SDK is a context variable holding the current span, a stack
discipline for nesting, and a writer. This module is exactly that:

  observe(name)        decorator — opens a span, nests it under the current
                       one, writes it as one row when the function returns
                       or raises
  current_trace_id()   trace id of the span currently open, or None
  annotate(**fields)   attach input/output/usage/metadata to the open span
  add_score(...)       attach an eval verdict to a trace as its own row
  read_traces()        read every row back

Spans land in `runs/traces.jsonl`, one JSON object per line, each carrying its
`trace_id`, `span_id`, `parent_id`, timing and whatever the call site attached.
A file is a deliberate choice over SQLite: traces are append-only, they are
read back in bulk, and a text file can be diffed and committed as evidence.
There is no hosted backend and no UI beyond `dashboard/app.py`.
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


def current_trace_id() -> str | None:
    """Trace id of the span currently open, or None outside any span."""
    span = _current.get()
    return span.trace_id if span else None


def annotate(**fields) -> None:
    """Attach fields to the span currently open; a no-op outside any span.

    `input` and `output` replace the span's own fields, `usage` is merged into
    its usage dict, `name` renames it, and every other key goes into
    `metadata`. There is no separate trace object: the root span *is* the
    trace, so trace-level fields are annotated onto the root span.
    """
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


def read_traces(path: Path = TRACE_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
