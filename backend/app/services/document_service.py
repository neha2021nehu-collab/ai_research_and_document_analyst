import uuid
from pathlib import Path
from typing import Any

import structlog

from backend.app.core.exceptions import DocumentProcessingError, ValidationError
from backend.app.services.chroma_service import chroma_service
from backend.app.services.ollama_service import ollama_service
from config.settings import settings

logger = structlog.get_logger(__name__)


class DocumentService:
    def __init__(self):
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def _validate_file(self, filename: str, size: int) -> None:
        if size > settings.max_file_size:
            raise ValidationError(
                f"File size {size} exceeds maximum allowed size {settings.max_file_size}"
            )
        ext = Path(filename).suffix.lower()
        if ext not in settings.allowed_extensions:
            raise ValidationError(
                f"File extension {ext} not allowed. Allowed: {settings.allowed_extensions}"
            )

    def _extract_text(self, file_path: str, content_type: str) -> str:
        ext = Path(file_path).suffix.lower()
        try:
            if ext == ".txt" or ext == ".md":
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
            elif ext == ".pdf":
                return self._extract_pdf(file_path)
            elif ext == ".docx":
                return self._extract_docx(file_path)
            else:
                raise ValidationError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error("Failed to extract text", file=file_path, error=str(e))
            raise DocumentProcessingError(f"Failed to extract text: {e}")

    def _extract_pdf(self, file_path: str) -> str:
        try:
            import pypdf
            text = ""
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise DocumentProcessingError("pypdf not installed. Cannot process PDF files.")
        except Exception as e:
            raise DocumentProcessingError(f"PDF extraction failed: {e}")

    def _extract_docx(self, file_path: str) -> str:
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise DocumentProcessingError("python-docx not installed. Cannot process DOCX files.")
        except Exception as e:
            raise DocumentProcessingError(f"DOCX extraction failed: {e}")

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self._chunk_size - self._chunk_overlap

        return chunks

    async def process_document(
        self,
        file_path: str,
        filename: str,
        content_type: str,
        size: int,
    ) -> dict[str, Any]:
        self._validate_file(filename, size)

        document_id = str(uuid.uuid4())
        text = self._extract_text(file_path, content_type)

        if not text.strip():
            raise DocumentProcessingError("Document contains no extractable text")

        chunks = self._chunk_text(text)
        chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]

        embeddings = await ollama_service.embed(chunks)

        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "content_type": content_type,
            }
            for i in range(len(chunks))
        ]

        await chroma_service.add_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids,
            embeddings=embeddings,
        )

        logger.info("Document processed successfully", document_id=document_id, chunks=len(chunks))

        return {
            "id": document_id,
            "filename": filename,
            "chunks_count": len(chunks),
            "status": "completed",
        }

    async def delete_document(self, document_id: str) -> None:
        try:
            results = await chroma_service.query(
                query_texts=[""],
                n_results=10000,
                where={"document_id": document_id},
            )
            ids = results.get("ids", [[]])[0]
            if ids:
                await chroma_service.delete_documents(ids)
            logger.info("Document deleted", document_id=document_id)
        except Exception as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            raise DocumentProcessingError(f"Failed to delete document: {e}")


document_service = DocumentService()
