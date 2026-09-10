from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentBase(BaseModel):
    filename: str
    content_type: str
    size: int


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: str
    status: DocumentStatus
    chunks_count: int = 0
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=5, ge=1, le=20)
    include_citations: bool = True


class Citation(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    model_used: str
    processing_time_ms: int


class SummaryRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1)
    max_length: int | None = Field(default=500, ge=100, le=2000)


class SummaryResponse(BaseModel):
    summary: str
    document_ids: list[str]
    model_used: str


class ResearchStep(BaseModel):
    step: int
    action: str
    query: str
    result: str
    citations: list[Citation] = []


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    max_steps: int | None = Field(default=5, ge=1, le=10)
    max_sources_per_step: int | None = Field(default=3, ge=1, le=5)


class ResearchResponse(BaseModel):
    topic: str
    steps: list[ResearchStep]
    final_answer: str
    all_citations: list[Citation]
    model_used: str
    total_processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict
