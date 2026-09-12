import structlog
from fastapi import APIRouter
from typing import Any

from backend.app.models import HealthResponse
from backend.app.services import chroma_service, ollama_service
from config.settings import settings

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    services: dict[str, dict[str, Any]] = {}

    try:
        info = await chroma_service.get_collection_info()
        services["chromadb"] = {"status": "healthy", "details": info}
    except Exception as e:
        services["chromadb"] = {"status": "unhealthy", "error": str(e)}

    try:
        models = await ollama_service.list_models()
        services["ollama"] = {"status": "healthy", "models_count": len(models)}
    except Exception as e:
        services["ollama"] = {"status": "unhealthy", "error": str(e)}

    all_healthy = all(s.get("status") == "healthy" for s in services.values())
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        services=services,
    )
