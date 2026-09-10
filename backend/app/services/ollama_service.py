from collections.abc import AsyncGenerator
from typing import Any

import ollama
import structlog

from backend.app.core.exceptions import LLMError
from config.settings import settings

logger = structlog.get_logger(__name__)


class OllamaService:
    def __init__(self):
        self._client: ollama.AsyncClient | None = None

    async def connect(self) -> None:
        try:
            self._client = ollama.AsyncClient(
                host=f"http://{settings.ollama_host}:{settings.ollama_port}"
            )
            await self._client.list()
            logger.info("Connected to Ollama", host=settings.ollama_host, port=settings.ollama_port)
        except Exception as e:
            logger.error("Failed to connect to Ollama", error=str(e))
            raise LLMError(f"Failed to connect to Ollama: {e}")

    async def disconnect(self) -> None:
        self._client = None

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise LLMError("Ollama not connected. Call connect() first.")

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        self._ensure_connected()
        try:
            model = model or settings.ollama_model
            response = await self._client.generate(
                model=model,
                prompt=prompt,
                system=system,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or -1,
                },
            )
            return response["response"]
        except Exception as e:
            logger.error("Failed to generate response from Ollama", error=str(e))
            raise LLMError(f"Failed to generate response: {e}")

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        self._ensure_connected()
        try:
            model = model or settings.ollama_model
            stream = await self._client.generate(
                model=model,
                prompt=prompt,
                system=system,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or -1,
                },
                stream=True,
            )
            async for chunk in stream:
                if chunk.get("response"):
                    yield chunk["response"]
        except Exception as e:
            logger.error("Failed to stream response from Ollama", error=str(e))
            raise LLMError(f"Failed to stream response: {e}")

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        self._ensure_connected()
        try:
            model = model or settings.ollama_embedding_model
            response = await self._client.embed(model=model, input=texts)
            return response["embeddings"]
        except Exception as e:
            logger.error("Failed to generate embeddings from Ollama", error=str(e))
            raise LLMError(f"Failed to generate embeddings: {e}")

    async def list_models(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        try:
            response = await self._client.list()
            return response.get("models", [])
        except Exception as e:
            logger.error("Failed to list Ollama models", error=str(e))
            raise LLMError(f"Failed to list models: {e}")

    async def pull_model(self, model: str) -> dict[str, Any]:
        self._ensure_connected()
        try:
            return await self._client.pull(model=model)
        except Exception as e:
            logger.error("Failed to pull Ollama model", model=model, error=str(e))
            raise LLMError(f"Failed to pull model {model}: {e}")


ollama_service = OllamaService()
