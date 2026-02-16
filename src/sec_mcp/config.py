"""Configuration management via environment variables.

Reads from .env file (via pydantic-settings) with sensible defaults.
All values can be overridden via environment variables.

Required:
    EDGAR_IDENTITY  — Your name + email for SEC EDGAR API User-Agent header

Optional:
    ANTHROPIC_API_KEY  — For Claude-powered narrative explanations
    MONGODB_URI        — For persistent filing cache in MongoDB Atlas
    PORT               — Server port (Railway sets this automatically)
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # SEC EDGAR API identity (name + email, required by SEC)
    edgar_identity: str = "SEC-MCP sec-mcp@example.com"

    # NLP model names (only used when running locally, not on Railway)
    sentiment_model: str = "ProsusAI/finbert"
    summarization_model: str = "facebook/bart-large-cnn"
    ner_model: str = "dslim/bert-base-NER"
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 128

    # Claude API for narrative explanations (optional)
    anthropic_api_key: str = ""

    # MongoDB for persistent filing cache (optional)
    mongodb_uri: str = ""

    # Server port (Railway sets PORT env var automatically)
    port: int = 8877

    # Strip whitespace from string fields — the .env file often has
    # trailing spaces that break connection strings
    @field_validator("mongodb_uri", "anthropic_api_key", "edgar_identity", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().strip('"').strip("'").strip()
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


_config: Settings | None = None


def get_config() -> Settings:
    """Get or create the shared Settings singleton."""
    global _config
    if _config is None:
        _config = Settings()
    return _config
