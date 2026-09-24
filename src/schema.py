from pydantic import BaseModel
from .cost import RequestCost


class Document(BaseModel):
    content: str
    score: float
    source: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[Document]
    trace_id: str | None = None
    cost: RequestCost | None = None

    model_config = {"arbitrary_types_allowed": True}
