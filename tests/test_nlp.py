"""Tests for NLP modules."""

import pytest

from sec_mcp.nlp.sentiment import SentimentAnalyzer
from sec_mcp.nlp.summarizer import FilingSummarizer
from sec_mcp.nlp.ner import EntityExtractor, MONEY_PATTERN, PERCENT_PATTERN, FISCAL_DATE_PATTERN


# --- Regex tests (no model loading needed) ---


def test_money_pattern():
    text = "The company paid $3.5 billion for the acquisition and $200 million in fees."
    matches = MONEY_PATTERN.findall(text)
    assert len(matches) == 2
    assert "$3.5 billion" in matches
    assert "$200 million" in matches


def test_percent_pattern():
    text = "Revenue grew 25.3% year over year while margins declined 2%."
    matches = PERCENT_PATTERN.findall(text)
    assert len(matches) == 2


def test_fiscal_date_pattern():
    text = "In Q3 2024, the company reported results. Fiscal year 2023 was strong. FY24 outlook is positive."
    matches = FISCAL_DATE_PATTERN.findall(text)
    assert len(matches) >= 2


# --- Model tests (slow, require model download) ---


@pytest.mark.slow
def test_positive_sentiment():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(
        "Revenue increased 25% year over year with strong margin expansion and record earnings."
    )
    assert result.overall_label == "positive"
    assert result.overall_score > 0.5
    assert result.num_chunks >= 1


@pytest.mark.slow
def test_negative_sentiment():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(
        "The company reported significant losses, declining revenue, and increased debt obligations."
    )
    assert result.overall_label == "negative"
    assert result.overall_score > 0.5


@pytest.mark.slow
def test_summarization():
    summarizer = FilingSummarizer()
    long_text = (
        "Apple Inc. reported record quarterly revenue of $123.9 billion for the fiscal "
        "first quarter ended December 30, 2023, up 2 percent year over year. The Company "
        "posted quarterly earnings per diluted share of $2.18, up 16 percent year over year "
        "during the quarter. International revenue accounted for 58 percent of total revenue. "
        "The board of directors has declared a cash dividend of $0.24 per share of common stock "
        "payable on February 15, 2024. Revenue from Products was $96.5 billion and revenue "
        "from Services reached an all-time record of $23.1 billion. The company also announced "
        "that it has returned over $25 billion to shareholders during the quarter."
    )
    result = summarizer.summarize(long_text)
    assert len(result.summary) > 0
    assert result.summary_length < result.original_length
    assert result.num_chunks_processed >= 1


@pytest.mark.slow
def test_entity_extraction():
    extractor = EntityExtractor()
    text = (
        "Apple Inc. acquired Beats Electronics from Dr. Dre for $3 billion in 2014. "
        "Tim Cook announced the deal at the Cupertino headquarters."
    )
    result = extractor.extract(text)
    assert len(result.entities) > 0

    labels = {e.label for e in result.entities}
    # Should find at least ORG and PER entities
    assert "ORG" in labels or "PER" in labels

    # Should find money via regex
    money_entities = [e for e in result.entities if e.label == "MONEY"]
    assert len(money_entities) >= 1
