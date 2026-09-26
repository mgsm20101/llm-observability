"""Response models — Document and RAGResponse in src/rag_observed.py."""
import pytest
from pydantic import ValidationError

from src.cost import calc_cost
from src.rag_observed import Document, RAGResponse


def test_document_accepts_valid_fields():
    doc = Document(content="leave policy text", score=0.87, source="leave-annual")

    assert doc.content == "leave policy text"
    assert doc.score == pytest.approx(0.87)
    assert doc.source == "leave-annual"


def test_document_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        Document(content="text", score=0.5)  # missing `source`


def test_document_coerces_numeric_string_score():
    doc = Document(content="text", score="0.5", source="doc-1")

    assert doc.score == pytest.approx(0.5)


def test_rag_response_defaults_trace_id_and_cost_to_none():
    response = RAGResponse(answer="21 days", sources=[])

    assert response.trace_id is None
    assert response.cost is None


def test_rag_response_holds_a_list_of_documents_and_a_cost():
    docs = [Document(content="t", score=0.9, source="leave-annual")]
    cost = calc_cost(
        input_tokens=100,
        output_tokens=50,
        price_input_per_mtok=0.0,
        price_output_per_mtok=0.0,
        model_name="gemma3:4b",
    )

    response = RAGResponse(answer="21 days", sources=docs, trace_id="abc123", cost=cost)

    assert response.sources[0].source == "leave-annual"
    assert response.cost.total_tokens == 150
    assert response.trace_id == "abc123"


def test_rag_response_rejects_non_list_sources():
    with pytest.raises(ValidationError):
        RAGResponse(answer="21 days", sources="not-a-list")
