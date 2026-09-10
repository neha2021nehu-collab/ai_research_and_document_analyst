from typing import Any

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from backend.app.core.exceptions import VectorDBError
from config.settings import settings

logger = structlog.get_logger(__name__)


class ChromaService:
    def __init__(self):
        self._client: chromadb.HttpClient | None = None
        self._collection = None

    async def connect(self) -> None:
        try:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Connected to ChromaDB",
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        except Exception as e:
            logger.error("Failed to connect to ChromaDB", error=str(e))
            raise VectorDBError(f"Failed to connect to ChromaDB: {e}") from e

    async def disconnect(self) -> None:
        self._client = None
        self._collection = None

    def _ensure_connected(self) -> None:
        if self._collection is None:
            raise VectorDBError("ChromaDB not connected. Call connect() first.")

    async def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self._ensure_connected()
        try:
            if embeddings:
                self._collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings,
                )
            else:
                self._collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
            logger.info("Documents added to ChromaDB", count=len(documents))
        except Exception as e:
            logger.error("Failed to add documents to ChromaDB", error=str(e))
            raise VectorDBError(f"Failed to add documents: {e}") from e

    async def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        self._ensure_connected()
        try:
            include = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")

            results = self._collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=include,
            )
            return results
        except Exception as e:
            logger.error("Failed to query ChromaDB", error=str(e))
            raise VectorDBError(f"Failed to query: {e}") from e

    async def delete_documents(self, ids: list[str]) -> None:
        self._ensure_connected()
        try:
            self._collection.delete(ids=ids)
            logger.info("Documents deleted from ChromaDB", count=len(ids))
        except Exception as e:
            logger.error("Failed to delete documents from ChromaDB", error=str(e))
            raise VectorDBError(f"Failed to delete documents: {e}") from e

    async def get_collection_info(self) -> dict[str, Any]:
        self._ensure_connected()
        try:
            count = self._collection.count()
            return {"name": settings.chroma_collection, "count": count}
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            raise VectorDBError(f"Failed to get collection info: {e}") from e


chroma_service = ChromaService()
