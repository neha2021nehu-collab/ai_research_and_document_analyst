import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
import structlog
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from backend.app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    LLMError,
    ValidationError,
    VectorDBError,
)
from backend.app.models import (
    DocumentResponse,
    DocumentStatus,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    ResearchResponse,
    SummaryRequest,
    SummaryResponse,
)
from backend.app.services import document_service, query_service
from config.settings import settings

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    try:
        content = await file.read()
        size = len(content)

        file_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{ext}"

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        result = await document_service.process_document(
            file_path=str(file_path),
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            size=size,
        )

        file_path.unlink()

        return DocumentResponse(
            id=result["id"],
            filename=result["filename"],
            content_type=file.content_type or "application/octet-stream",
            size=size,
            status=DocumentStatus.COMPLETED,
            chunks_count=result["chunks_count"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except DocumentProcessingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error("Upload failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    try:
        from backend.app.services.chroma_service import chroma_service

        results = await chroma_service.query(
            query_texts=[""],
            n_results=1,
            where={"document_id": document_id},
        )
        if not results.get("ids", [[]])[0]:
            raise DocumentNotFoundError(document_id)

        meta = results["metadatas"][0][0]
        return DocumentResponse(
            id=document_id,
            filename=meta.get("filename", "unknown"),
            content_type=meta.get("content_type", "unknown"),
            size=0,
            status=DocumentStatus.COMPLETED,
            chunks_count=len(results.get("ids", [[]])[0]),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error("Get document failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        await document_service.delete_document(document_id)
        return {"message": "Document deleted successfully"}
    except DocumentProcessingError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error("Delete document failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        return await query_service.query(request)
    except (VectorDBError, LLMError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    try:
        summary = await query_service.summarize(request.document_ids, request.max_length)
        return SummaryResponse(
            summary=summary,
            document_ids=request.document_ids,
            model_used=settings.ollama_model,
        )
    except (VectorDBError, LLMError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error("Summarize failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    try:
        return await query_service.research(request)
    except (VectorDBError, LLMError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error("Research failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e
