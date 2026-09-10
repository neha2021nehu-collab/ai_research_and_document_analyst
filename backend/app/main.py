from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import api_router, health_router
from backend.app.core.logging import get_logger, setup_logging
from backend.app.services import chroma_service, ollama_service
from config.settings import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting AI Research Assistant", version=settings.app_version)

    try:
        await chroma_service.connect()
        await ollama_service.connect()
        logger.info("All services connected successfully")
    except Exception as e:
        logger.error("Failed to connect to services", error=str(e))

    yield

    logger.info("Shutting down AI Research Assistant")
    await chroma_service.disconnect()
    await ollama_service.disconnect()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Research Assistant with Document Ingestion, RAG, and Agentic Research",
    lifespan=lifespan,
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

app.include_router(health_router)
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
