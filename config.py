from __future__ import annotations
from pydantic import BaseModel, Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    
    """Configuration for the RAG pipeline."""
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_model: str = Field(..., env="OPENROUTER_MODEL")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL")
    db_dir: str = Field("db", env="DB_DIR")
    retrieval_k: int = Field(5, env="RETRIEVAL_K")
    temperature: float = Field(0.7, env="TEMPERATURE")
    chunk_size: int = Field(1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    request_timeout: int = Field(90, env="REQUEST_TIMEOUT")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")


    # -------- Meta / SettingsConfigDict ----------------------------------------
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        validate_assignment = True,
        extra = "forbid",
        case_sensitive = False,
    )
