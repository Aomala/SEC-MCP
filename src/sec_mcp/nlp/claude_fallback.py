"""Claude-based NLP fallback for when BERT/BART models are unavailable.

Provides the same interfaces as sentiment.py, summarizer.py, and ner.py
but uses the Anthropic Claude API instead of local transformer models.
This enables NLP tools to work in production without ~2.5GB of model weights.
"""

from __future__ import annotations

import json
import logging
import re

from sec_mcp.models import (
    ChunkSentiment,
    Entity,
    EntityExtractionResult,
    SentimentAnalysis,
    SummaryResult,
)

log = logging.getLogger(__name__)

_anthropic_client = None


def _get_client():
    """Get or create the shared Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        from sec_mcp.config import get_config
        config = get_config()
        if not config.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for Claude-based NLP fallback. "
                "Set it in .env or install transformers+torch for local models."
            )
        _anthropic_client = Anthropic(api_key=config.anthropic_api_key)
    return _anthropic_client


class ClaudeSentimentAnalyzer:
    """Sentiment analyzer using Claude API instead of FinBERT."""

    def analyze(self, text: str) -> SentimentAnalysis:
        """Analyze financial sentiment using Claude."""
        client = _get_client()

        # Truncate to avoid excessive token usage
        truncated = text[:8000]

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": (
                    "Analyze the financial sentiment of the following text. "
                    "Respond ONLY with a JSON object: "
                    '{"label": "positive"|"negative"|"neutral", "score": 0.0-1.0, '
                    '"reasoning": "brief explanation"}\n\n'
                    f"Text:\n{truncated}"
                ),
            }],
        )

        try:
            result = json.loads(response.content[0].text)
            label = result.get("label", "neutral").lower()
            score = float(result.get("score", 0.5))
        except (json.JSONDecodeError, IndexError, KeyError):
            label = "neutral"
            score = 0.5

        return SentimentAnalysis(
            overall_label=label,
            overall_score=score,
            chunk_results=[ChunkSentiment(chunk_index=0, label=label, score=score)],
            num_chunks=1,
        )


class ClaudeFilingSummarizer:
    """Summarizer using Claude API instead of BART."""

    def summarize(
        self,
        text: str,
        max_summary_length: int = 300,
        min_summary_length: int = 50,
    ) -> SummaryResult:
        """Summarize text using Claude."""
        client = _get_client()

        original_length = len(text)
        # Truncate to avoid excessive token usage
        truncated = text[:15000]

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize the following SEC filing text in {max_summary_length} words or fewer. "
                    "Focus on key financial metrics, risks, and business developments. "
                    "Be concise and factual.\n\n"
                    f"Text:\n{truncated}"
                ),
            }],
        )

        summary = response.content[0].text.strip()

        return SummaryResult(
            summary=summary,
            original_length=original_length,
            summary_length=len(summary),
            num_chunks_processed=1,
        )


class ClaudeEntityExtractor:
    """Entity extractor using Claude API instead of BERT NER."""

    # Keep regex patterns for financial entities (cheap, no API call needed)
    _MONEY_PATTERN = re.compile(
        r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|mn|bn|k))?",
        re.IGNORECASE,
    )
    _PERCENT_PATTERN = re.compile(r"\d+\.?\d*\s*%")
    _FISCAL_DATE_PATTERN = re.compile(
        r"(?:Q[1-4]\s*\d{4}|fiscal\s+year\s+\d{4}|FY\s*\d{2,4}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> EntityExtractionResult:
        """Extract entities using Claude + regex."""
        client = _get_client()

        # Regex extraction (free, always works)
        entities: list[Entity] = []
        seen_texts: set[str] = set()

        for pattern, label in [
            (self._MONEY_PATTERN, "MONEY"),
            (self._PERCENT_PATTERN, "PERCENT"),
            (self._FISCAL_DATE_PATTERN, "DATE"),
        ]:
            for match in pattern.finditer(text):
                t = match.group().strip()
                if t not in seen_texts:
                    seen_texts.add(t)
                    entities.append(Entity(text=t, label=label, score=1.0))

        # Claude extraction for ORG, PER, LOC
        truncated = text[:8000]

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract named entities from the following SEC filing text. "
                        "Return ONLY a JSON array of objects: "
                        '[{"text": "entity name", "label": "ORG"|"PER"|"LOC"}]\n'
                        "Include companies, people, and locations. Deduplicate.\n\n"
                        f"Text:\n{truncated}"
                    ),
                }],
            )

            claude_entities = json.loads(response.content[0].text)
            for e in claude_entities:
                t = e.get("text", "").strip()
                if t and t not in seen_texts:
                    seen_texts.add(t)
                    entities.append(Entity(
                        text=t,
                        label=e.get("label", "ORG"),
                        score=0.9,
                    ))
        except (json.JSONDecodeError, Exception) as exc:
            log.warning("Claude NER extraction failed: %s", exc)

        # Build entity counts
        counts: dict[str, int] = {}
        for e in entities:
            counts[e.label] = counts.get(e.label, 0) + 1

        return EntityExtractionResult(entities=entities, entity_counts=counts)
