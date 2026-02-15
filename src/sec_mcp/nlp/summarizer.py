"""BART-based summarization for SEC filing sections."""

from __future__ import annotations

from sec_mcp.config import get_config
from sec_mcp.models import SummaryResult
from sec_mcp.nlp.chunker import chunk_text


class FilingSummarizer:
    """Lazy-loaded BART summarizer with hierarchical summarization for long docs."""

    def __init__(self, model_name: str | None = None):
        self._pipeline = None
        self._tokenizer = None
        self._model_name = model_name or get_config().summarization_model

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._pipeline = pipeline(
                "summarization",
                model=self._model_name,
                tokenizer=self._tokenizer,
            )

    def summarize(
        self,
        text: str,
        max_summary_length: int = 300,
        min_summary_length: int = 50,
    ) -> SummaryResult:
        """Summarize text with hierarchical chunking for long documents."""
        self._load()

        original_length = len(text)

        # BART handles up to 1024 tokens
        chunks = chunk_text(
            text,
            self._tokenizer,
            max_tokens=1024,
            overlap_tokens=128,
        )

        if not chunks:
            return SummaryResult(
                summary="",
                original_length=original_length,
                summary_length=0,
                num_chunks_processed=0,
            )

        # First pass: summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            result = self._pipeline(
                chunk,
                max_length=max_summary_length,
                min_length=min(min_summary_length, len(chunk.split()) // 2),
                do_sample=False,
            )
            chunk_summaries.append(result[0]["summary_text"])

        num_chunks = len(chunks)

        # If multiple chunks, do a second pass on concatenated summaries
        if len(chunk_summaries) > 1:
            combined = " ".join(chunk_summaries)
            combined_tokens = self._tokenizer.encode(combined, add_special_tokens=False)

            if len(combined_tokens) > 1024:
                # Recursively summarize the summaries
                return self.summarize(combined, max_summary_length, min_summary_length)

            result = self._pipeline(
                combined,
                max_length=max_summary_length,
                min_length=min(min_summary_length, len(combined.split()) // 2),
                do_sample=False,
            )
            final_summary = result[0]["summary_text"]
        else:
            final_summary = chunk_summaries[0]

        return SummaryResult(
            summary=final_summary,
            original_length=original_length,
            summary_length=len(final_summary),
            num_chunks_processed=num_chunks,
        )
