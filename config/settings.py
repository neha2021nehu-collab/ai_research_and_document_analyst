
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Research Assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    frontend_host: str = "0.0.0.0"
    frontend_port: int = 8501

    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "documents"

    ollama_host: str = "ollama"
    ollama_port: int = 11434
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    embedding_dim: int = 768
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    max_file_size: int = 50 * 1024 * 1024
    allowed_extensions: list[str] = [".pdf", ".txt", ".md", ".docx"]

    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    rate_limit_requests: int = 100
    rate_limit_window: int = 60


settings = Settings()
