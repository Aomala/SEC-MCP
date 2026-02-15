"""Token-aware text chunking for long documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 512,
    overlap_tokens: int = 128,
) -> list[str]:
    """Split text into overlapping chunks based on token count.

    Uses the model's tokenizer for accurate splitting so chunks
    don't exceed the model's context window.
    """
    if not text or not text.strip():
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)

    if total_tokens <= max_tokens:
        return [text]

    chunks = []
    start = 0
    step = max_tokens - overlap_tokens

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_ids = token_ids[start:end]
        chunk_text_decoded = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text_decoded.strip():
            chunks.append(chunk_text_decoded)
        if end >= total_tokens:
            break
        start += step

    return chunks
