"""A small RAG pipeline, instrumented end to end.

Three spans — retrieve, rerank, generate — under one trace, each carrying its
own timing, inputs and outputs, and the generation span carrying real token
counts read back from Ollama. The traces land in `runs/traces.jsonl` via
`tracer.py`; there is no hosted backend and no account.

The retrieval quality here is not the point and is not measured. What is being
demonstrated is that every stage of an LLM call is attributable: how long it
took, what it was given, what it returned, and what it would have cost.
"""
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel

from .config import get_settings
from .cost import RequestCost, calc_cost
from .store import DenseStore, load_corpus
from .tracer import annotate, current_trace_id, observe

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# The reranker was `cross-encoder/ms-marco-MiniLM-L-6-v2`, which is trained on
# English MS MARCO and was being asked to score Arabic passages. The
# multilingual mMARCO variant is at least trained on the right languages, and
# it is already in the local HF cache so it costs no download.
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Document(BaseModel):
    """One retrieved passage; `source` is the corpus document id."""

    content: str
    score: float
    source: str


class RAGResponse(BaseModel):
    """What `answer_question` returns: the answer, its sources, its trace."""

    answer: str
    sources: list[Document]
    trace_id: str | None = None
    cost: RequestCost | None = None

    model_config = {"arbitrary_types_allowed": True}


_settings = get_settings()
_store: DenseStore | None = None
_reranker: "CrossEncoder | None" = None


def _get_store() -> DenseStore:
    global _store
    if _store is None:
        _store = DenseStore(
            load_corpus(_settings.corpus_path), _settings.embed_model
        )
    return _store


def _get_reranker() -> "CrossEncoder":
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# ──────────────────────────────────────────────
# Main pipeline entry point — the root span
# ──────────────────────────────────────────────

@observe(name="rag_pipeline")
def answer_question(query: str) -> RAGResponse:
    """Answer an Arabic HR-policy question from the local corpus.

    Opens the root span `rag_pipeline` (the trace) with three child spans:
        retrieve → rerank → generate
    Every span is written to `runs/traces.jsonl` as it closes.
    """
    docs = _retrieve(query)
    docs_reranked = _rerank(query, docs)
    answer, input_tok, output_tok = _generate(query, docs_reranked)

    cost = calc_cost(
        input_tokens=input_tok,
        output_tokens=output_tok,
        price_input_per_mtok=_settings.price_input_per_mtok,
        price_output_per_mtok=_settings.price_output_per_mtok,
        model_name=_settings.ollama_model,
    )

    trace_id = current_trace_id()

    annotate(
        input=query,
        output=answer,
        tags=["p7-observability", "hr-policy"],
        metadata={
            "cost_usd": cost.cost_usd,
            "total_tokens": cost.total_tokens,
        },
    )

    return RAGResponse(
        answer=answer,
        sources=docs_reranked,
        trace_id=trace_id,
        cost=cost,
    )


# ──────────────────────────────────────────────
# Span 1 — Vector retrieval
# ──────────────────────────────────────────────

@observe(name="retrieve")
def _retrieve(query: str, top_k: int = 8) -> list[Document]:
    """Dense retrieval from the in-process store."""
    hits = _get_store().search(query, top_k)

    docs = [
        Document(content=f"{doc.title}\n{doc.text}", score=score, source=doc.id)
        for doc, score in hits
    ]

    annotate(
        input={"query": query, "top_k": top_k},
        output={"doc_ids": [d.source for d in docs]},
        metadata={"returned_docs": len(docs), "encoder": _settings.embed_model},
    )
    return docs


# ──────────────────────────────────────────────
# Span 2 — Cross-encoder reranking
# ──────────────────────────────────────────────

@observe(name="rerank")
def _rerank(query: str, docs: list[Document], top_k: int | None = None) -> list[Document]:
    """Rerank candidates using a cross-encoder."""
    if not docs:
        return []

    top_k = _settings.top_k if top_k is None else top_k
    reranker = _get_reranker()
    pairs = [(query, doc.content) for doc in docs]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(docs, scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    result = [Document(content=d.content, score=float(s), source=d.source)
              for d, s in reranked]

    annotate(
        input={"input_docs": len(docs), "model": RERANK_MODEL},
        output={"output_docs": len(result), "doc_ids": [d.source for d in result]},
    )
    return result


# ──────────────────────────────────────────────
# Span 3 — LLM generation (as_type="generation" → token tracking)
# ──────────────────────────────────────────────

@observe(name="generate", as_type="generation")
def _generate(query: str, docs: list[Document]) -> tuple[str, int, int]:
    """Generate answer via Ollama. Returns (answer, input_tokens, output_tokens)."""
    context = "\n\n".join(f"[{i+1}] {d.content}" for i, d in enumerate(docs))

    prompt = (
        "أنت مساعد للموارد البشرية.\n"
        "استخدم فقط النصوص التالية للإجابة. إذا لم تجد الإجابة، قل 'لا تتوفر معلومات كافية'.\n\n"
        f"النصوص:\n{context}\n\n"
        f"السؤال: {query}\n\n"
        "الإجابة:"
    )

    response = httpx.post(
        f"{_settings.ollama_base_url}/api/chat",
        json={
            "model": _settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=300.0,
    ).raise_for_status().json()

    answer = response["message"]["content"]
    input_tok = response.get("prompt_eval_count", 0)
    output_tok = response.get("eval_count", 0)

    annotate(
        model=_settings.ollama_model,
        usage={"input": input_tok, "output": output_tok},
    )
    return answer, input_tok, output_tok
