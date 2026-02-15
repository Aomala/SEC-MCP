"""Named entity recognition with BERT + regex for financial entities."""

from __future__ import annotations

import re

from sec_mcp.config import get_config
from sec_mcp.models import Entity, EntityExtractionResult
from sec_mcp.nlp.chunker import chunk_text

# Regex patterns for financial entities that BERT NER doesn't catch
MONEY_PATTERN = re.compile(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|mn|bn|k))?", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"\d+\.?\d*\s*%")
FISCAL_DATE_PATTERN = re.compile(
    r"(?:Q[1-4]\s*\d{4}|fiscal\s+year\s+\d{4}|FY\s*\d{2,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


class EntityExtractor:
    """Lazy-loaded NER using dslim/bert-base-NER + regex for financial patterns."""

    def __init__(self, model_name: str | None = None):
        self._pipeline = None
        self._tokenizer = None
        self._model_name = model_name or get_config().ner_model

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._pipeline = pipeline(
                "ner",
                model=self._model_name,
                tokenizer=self._tokenizer,
                aggregation_strategy="simple",
            )

    def _extract_regex_entities(self, text: str) -> list[Entity]:
        """Extract financial entities via regex patterns."""
        entities = []

        for match in MONEY_PATTERN.finditer(text):
            entities.append(Entity(
                text=match.group().strip(),
                label="MONEY",
                score=1.0,
            ))

        for match in PERCENT_PATTERN.finditer(text):
            entities.append(Entity(
                text=match.group().strip(),
                label="PERCENT",
                score=1.0,
            ))

        for match in FISCAL_DATE_PATTERN.finditer(text):
            entities.append(Entity(
                text=match.group().strip(),
                label="DATE",
                score=1.0,
            ))

        return entities

    def extract(self, text: str) -> EntityExtractionResult:
        """Extract named entities from text using BERT NER + regex."""
        self._load()

        config = get_config()
        chunks = chunk_text(
            text,
            self._tokenizer,
            max_tokens=config.max_chunk_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
        )

        # BERT NER on chunks
        bert_entities: list[Entity] = []
        seen_texts: set[str] = set()

        for chunk in chunks:
            results = self._pipeline(chunk)
            for r in results:
                entity_text = r["word"].strip()
                if entity_text and entity_text not in seen_texts:
                    seen_texts.add(entity_text)
                    bert_entities.append(Entity(
                        text=entity_text,
                        label=r["entity_group"],
                        score=round(r["score"], 4),
                    ))

        # Regex entities on full text (no chunking needed)
        regex_entities = self._extract_regex_entities(text)

        # Deduplicate regex entities
        for entity in regex_entities:
            if entity.text not in seen_texts:
                seen_texts.add(entity.text)
                bert_entities.append(entity)

        all_entities = bert_entities

        # Count by label
        counts: dict[str, int] = {}
        for e in all_entities:
            counts[e.label] = counts.get(e.label, 0) + 1

        return EntityExtractionResult(
            entities=all_entities,
            entity_counts=counts,
        )
