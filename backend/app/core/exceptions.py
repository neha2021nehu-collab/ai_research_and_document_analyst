from typing import Any


class AppException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Any | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class DocumentNotFoundError(AppException):
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document not found: {document_id}",
            status_code=404,
            detail={"document_id": document_id},
        )


class DocumentProcessingError(AppException):
    def __init__(self, message: str, detail: Any | None = None):
        super().__init__(
            message=f"Document processing failed: {message}",
            status_code=500,
            detail=detail,
        )


class VectorDBError(AppException):
    def __init__(self, message: str, detail: Any | None = None):
        super().__init__(
            message=f"Vector database error: {message}",
            status_code=503,
            detail=detail,
        )


class LLMError(AppException):
    def __init__(self, message: str, detail: Any | None = None):
        super().__init__(
            message=f"LLM service error: {message}",
            status_code=503,
            detail=detail,
        )


class ValidationError(AppException):
    def __init__(self, message: str, detail: Any | None = None):
        super().__init__(
            message=f"Validation error: {message}",
            status_code=400,
            detail=detail,
        )


class RateLimitError(AppException):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=429,
            detail={"retry_after": 60},
        )
