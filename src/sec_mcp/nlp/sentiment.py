"""FinBERT sentiment analysis for financial text."""

from __future__ import annotations

from sec_mcp.config import get_config
from sec_mcp.models import ChunkSentiment, SentimentAnalysis
from sec_mcp.nlp.chunker import chunk_text


class SentimentAnalyzer:
    """Lazy-loaded FinBERT sentiment analyzer."""

    def __init__(self, model_name: str | None = None):
        self._pipeline = None
        self._tokenizer = None
        self._model_name = model_name or get_config().sentiment_model

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                tokenizer=self._tokenizer,
                top_k=None,
            )

    def analyze(self, text: str) -> SentimentAnalysis:
        """Run sentiment analysis, chunking long text as needed."""
        self._load()

        config = get_config()
        chunks = chunk_text(
            text,
            self._tokenizer,
            max_tokens=config.max_chunk_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
        )

        if not chunks:
            return SentimentAnalysis(
                overall_label="neutral",
                overall_score=0.0,
                chunk_results=[],
                num_chunks=0,
            )

        # Aggregate scores across chunks
        label_scores: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        chunk_results: list[ChunkSentiment] = []
        total_weight = 0.0

        for i, chunk in enumerate(chunks):
            results = self._pipeline(chunk[:512])[0]  # pipeline returns list of list
            weight = len(chunk)
            total_weight += weight

            # Find best label for this chunk
            best = max(results, key=lambda x: x["score"])
            chunk_results.append(ChunkSentiment(
                chunk_index=i,
                label=best["label"],
                score=round(best["score"], 4),
            ))

            # Accumulate weighted scores
            for r in results:
                label = r["label"]
                if label in label_scores:
                    label_scores[label] += r["score"] * weight

        # Normalize
        if total_weight > 0:
            for label in label_scores:
                label_scores[label] /= total_weight

        overall_label = max(label_scores, key=label_scores.get)
        overall_score = round(label_scores[overall_label], 4)

        return SentimentAnalysis(
            overall_label=overall_label,
            overall_score=overall_score,
            chunk_results=chunk_results,
            num_chunks=len(chunks),
        )
