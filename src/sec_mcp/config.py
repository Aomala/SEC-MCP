"""Configuration management via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    edgar_identity: str = "SEC-MCP sec-mcp@example.com"
    sentiment_model: str = "ProsusAI/finbert"
    summarization_model: str = "facebook/bart-large-cnn"
    ner_model: str = "dslim/bert-base-NER"
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 128

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


_config: Settings | None = None


def get_config() -> Settings:
    global _config
    if _config is None:
        _config = Settings()
    return _config
