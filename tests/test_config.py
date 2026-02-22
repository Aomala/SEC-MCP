"""Tests for sec_mcp.config — Settings class and validators."""

import pytest

from sec_mcp.config import Settings


class TestSettingsDefaults:
    """Verify default values match the expected configuration.

    Uses _env_file=None + monkeypatch to isolate from the local .env file
    so we test hardcoded defaults, not whatever is in .env.
    """

    def _bare(self, monkeypatch) -> "Settings":
        """Return a Settings instance with no .env and no relevant env vars."""
        for var in ("EDGAR_IDENTITY", "ANTHROPIC_API_KEY", "MONGODB_URI"):
            monkeypatch.delenv(var, raising=False)
        return Settings(_env_file=None)

    def test_default_edgar_identity(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.edgar_identity == "SEC-MCP sec-mcp@example.com"

    def test_default_port(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.port == 8877

    def test_default_anthropic_api_key_empty(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.anthropic_api_key == ""

    def test_default_mongodb_uri_empty(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.mongodb_uri == ""

    def test_default_sentiment_model(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.sentiment_model == "ProsusAI/finbert"

    def test_default_summarization_model(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.summarization_model == "facebook/bart-large-cnn"

    def test_default_ner_model(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.ner_model == "dslim/bert-base-NER"

    def test_default_max_chunk_tokens(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.max_chunk_tokens == 512

    def test_default_chunk_overlap_tokens(self, monkeypatch):
        s = self._bare(monkeypatch)
        assert s.chunk_overlap_tokens == 128


class TestWhitespaceStripping:
    """Verify the strip_whitespace validator handles edge cases."""

    def test_strips_trailing_spaces(self):
        s = Settings(edgar_identity="  hello@example.com  ")
        assert s.edgar_identity == "hello@example.com"

    def test_strips_surrounding_double_quotes(self):
        s = Settings(mongodb_uri='"mongodb+srv://user:pass@host"')
        assert s.mongodb_uri == "mongodb+srv://user:pass@host"

    def test_strips_surrounding_single_quotes(self):
        s = Settings(anthropic_api_key="'sk-ant-12345'")
        assert s.anthropic_api_key == "sk-ant-12345"

    def test_strips_quotes_and_spaces_combined(self):
        s = Settings(edgar_identity='  "John Doe john@co.com"  ')
        assert s.edgar_identity == "John Doe john@co.com"

    def test_non_validated_field_unchanged(self):
        """Fields not listed in the validator should not be stripped."""
        s = Settings(sentiment_model="  ProsusAI/finbert  ")
        # sentiment_model is NOT in the validator list, so spaces remain
        assert s.sentiment_model == "  ProsusAI/finbert  "
