import pytest

from config.settings import Settings


class TestSettings:
    def test_default_settings(self):
        settings = Settings()
        assert settings.app_name == "AI Research Assistant"
        assert settings.api_port == 8000
        assert settings.frontend_port == 8501
        assert settings.chroma_port == 8000
        assert settings.ollama_port == 11434
        assert settings.ollama_model == "llama3.2"

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("APP_NAME", "Test App")
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("DEBUG", "true")

        settings = Settings()
        assert settings.app_name == "Test App"
        assert settings.api_port == 9000
        assert settings.debug is True


class TestModels:
    def test_document_status_enum(self):
        from backend.app.models import DocumentStatus

        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

    def test_query_request_validation(self):
        from backend.app.models import QueryRequest

        req = QueryRequest(question="Test question", top_k=3)
        assert req.question == "Test question"
        assert req.top_k == 3
        assert req.include_citations is True

    def test_query_request_invalid_top_k(self):
        from pydantic import ValidationError

        from backend.app.models import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(question="Test", top_k=0)

        with pytest.raises(ValidationError):
            QueryRequest(question="Test", top_k=21)


class TestExceptions:
    def test_app_exception(self):
        from backend.app.core.exceptions import AppException

        exc = AppException("Test error", status_code=400, detail={"key": "value"})
        assert exc.message == "Test error"
        assert exc.status_code == 400
        assert exc.detail == {"key": "value"}

    def test_document_not_found_error(self):
        from backend.app.core.exceptions import DocumentNotFoundError

        exc = DocumentNotFoundError("doc-123")
        assert exc.status_code == 404
        assert "doc-123" in exc.message
        assert exc.detail == {"document_id": "doc-123"}

    def test_validation_error(self):
        from backend.app.core.exceptions import ValidationError

        exc = ValidationError("Invalid input", detail={"field": "email"})
        assert exc.status_code == 400
        assert exc.detail == {"field": "email"}


class TestServices:
    @pytest.mark.asyncio
    async def test_chroma_service_connection(self):
        from backend.app.services.chroma_service import ChromaService

        service = ChromaService()
        assert service._client is None
        assert service._collection is None

    @pytest.mark.asyncio
    async def test_ollama_service_connection(self):
        from backend.app.services.ollama_service import OllamaService

        service = OllamaService()
        assert service._client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
