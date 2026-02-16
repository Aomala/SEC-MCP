"""10-K / 10-Q Section Segmenter — line-level boundary detection.

Inspired by the BERT4ItemSeg / GPT4ItemSeg paper (Lu et al.):
  "Utilizing Pre-trained and Large Language Models for 10-K Items Segmentation"

Key insight from the paper: item boundaries coincide with line breaks,
and items follow a known sequential order.  TOC entries can be
distinguished from actual section headers by position, context, and
structural cues.

Algorithm:
  1.  Split the filing plain-text into lines (with line IDs).
  2.  Find *all* candidate "Item N" headers.
  3.  Classify each as **TOC reference** vs **actual section header** using:
        – document-position (TOC in first ~15 %)
        – line length / presence of trailing page numbers
        – surrounding-context (followed by content vs another header)
  4.  Build a complete section-boundary map (item → start_line … end_line)
      enforcing the known item order.
  5.  Cache the result per accession so repeat requests are free.

If the deterministic pass fails, an optional LLM fallback uses
Line-ID-Based (LIB) prompting with Claude to identify boundaries.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Item metadata
# ═══════════════════════════════════════════════════════════════════════════

# Canonical item order for 10-K (SEC Regulation S-K)
_10K_ITEM_ORDER: list[str] = [
    "1", "1A", "1B", "1C", "2", "3", "4",
    "5", "6", "7", "7A", "8",
    "9", "9A", "9B",
    "10", "11", "12", "13", "14", "15", "16",
]

# Canonical item order for 10-Q
_10Q_ITEM_ORDER: list[str] = [
    "P1-1", "P1-2", "P1-3", "P1-4",
    "P2-1", "P2-1A", "P2-2", "P2-3", "P2-4", "P2-5", "P2-6",
]

# Friendly name → item-id mapping (shared single source of truth)
SECTION_ALIASES: dict[str, str] = {
    # 10-K
    "business":                  "1",
    "risk_factors":              "1A",
    "risk factors":              "1A",
    "risk factor":               "1A",
    "unresolved staff comments": "1B",
    "cybersecurity":             "1C",
    "properties":                "2",
    "legal":                     "3",
    "legal proceedings":         "3",
    "legal_proceedings":         "3",
    "mine safety":               "4",
    "mda":                       "7",
    "md&a":                      "7",
    "management discussion":     "7",
    "management's discussion":   "7",
    "discussion and analysis":   "7",
    "management_discussion":     "7",
    "quantitative":              "7A",
    "financial_statements":      "8",
    "financial statements":      "8",
    "controls":                  "9A",
    "controls and procedures":   "9A",
    "internal controls":         "9A",
    "executive compensation":    "11",
    "executive_compensation":    "11",
    "security ownership":        "12",
    "certain relationships":     "13",
    "accountant fees":           "14",
    "exhibits":                  "15",
    # 10-Q aliases
    "financial_statements_q":    "P1-1",
    "mda_q":                     "P1-2",
    "risk_factors_q":            "P2-1A",
}

# Title keywords per item (for secondary header matching)
_ITEM_TITLES: dict[str, re.Pattern[str]] = {
    "1":  re.compile(r"business", re.I),
    "1A": re.compile(r"risk\s+factors?", re.I),
    "1B": re.compile(r"unresolved\s+staff\s+comments?", re.I),
    "1C": re.compile(r"cybersecurity", re.I),
    "2":  re.compile(r"properties", re.I),
    "3":  re.compile(r"legal\s+proceedings?", re.I),
    "4":  re.compile(r"mine\s+safety", re.I),
    "5":  re.compile(r"market\s+for\s+(?:the\s+)?registrant", re.I),
    "6":  re.compile(r"(?:\[?reserved\]?|selected\s+financial)", re.I),
    "7":  re.compile(r"management.s?\s+discussion|md\s*&\s*a", re.I),
    "7A": re.compile(r"quantitative\s+and\s+qualitative", re.I),
    "8":  re.compile(r"financial\s+statements?\s+and\s+suppl", re.I),
    "9":  re.compile(r"changes?\s+in\s+and\s+disagreements?", re.I),
    "9A": re.compile(r"controls?\s+and\s+procedures?", re.I),
    "9B": re.compile(r"other\s+information", re.I),
    "10": re.compile(r"directors?,?\s+executive\s+officers?|corporate\s+governance", re.I),
    "11": re.compile(r"executive\s+compensation", re.I),
    "12": re.compile(r"security\s+ownership", re.I),
    "13": re.compile(r"certain\s+relationships?", re.I),
    "14": re.compile(r"principal\s+account(?:ant|ing)\s+fees?", re.I),
    "15": re.compile(r"exhibits?\s+(?:and\s+)?financial\s+statement\s+schedules?", re.I),
}

# Header regex: matches "Item 7", "ITEM 1A.", "Item 9A:", etc.
_ITEM_HDR_RE = re.compile(
    r"^[\s]*"
    r"(?:PART\s+[IVX]+[\s\.\:\-]*)??"    # optional "PART I"
    r"ITEM\s+"                             # "ITEM " or "Item "
    r"(\d+[A-Za-z]?)"                     # capture item number
    r"[\.\:\s\-\u2013\u2014]*"            # separator
    r"(.*)",                               # rest of line
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
#  SectionMap
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SectionMap:
    """Parsed filing with line-level section boundaries.

    ``sections`` maps canonical item ids (e.g. ``"7"``) to
    ``(start_line_inclusive, end_line_exclusive)`` in *self.lines*.
    """

    lines: list[str]
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)

    # ── public API ─────────────────────────────────────────────────────

    def get_section_text(self, item_id: str) -> str | None:
        """Return joined text for an item, or *None* if not mapped."""
        bounds = self.sections.get(item_id.upper())
        if not bounds:
            return None
        s, e = bounds
        raw = "\n".join(self.lines[s:e])
        cleaned = _clean_section_text(raw)

        # Handle "incorporated by reference" stubs (common for banks)
        # If the section is very short and references another location,
        # search the full document for the actual content.
        if cleaned and len(cleaned) < 1500 and _is_reference_stub(cleaned):
            actual = _find_actual_content(self.lines, item_id)
            if actual and len(actual) > len(cleaned):
                return actual

        return cleaned

    def get_section_by_name(self, name: str) -> str | None:
        """Resolve a friendly name and return the section text."""
        item_id = SECTION_ALIASES.get(name.lower().strip())
        if item_id is None:
            return None
        return self.get_section_text(item_id)

    def available_items(self) -> list[str]:
        return sorted(self.sections.keys(), key=_item_sort_key)

    def __repr__(self) -> str:
        items = ", ".join(f"Item {k}" for k in self.available_items())
        return f"<SectionMap {len(self.lines)} lines, items=[{items}]>"


# ═══════════════════════════════════════════════════════════════════════════
#  Deterministic segmenter
# ═══════════════════════════════════════════════════════════════════════════

def segment_filing(text: str, form_type: str = "10-K") -> SectionMap:
    """Segment a 10-K or 10-Q filing into sections.

    Returns a ``SectionMap`` with line-level section boundaries.
    """
    lines = text.split("\n")
    n_lines = len(lines)
    if n_lines == 0:
        return SectionMap(lines, {})

    is_10q = "10-Q" in form_type.upper()
    item_order = _10Q_ITEM_ORDER if is_10q else _10K_ITEM_ORDER

    # ── Pass 1: find ALL candidate Item headers ───────────────────────
    candidates: list[tuple[int, str, str]] = []  # (line_idx, item_id, rest)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _ITEM_HDR_RE.match(stripped)
        if m:
            item_id = m.group(1).upper()
            rest = m.group(2).strip()
            # Validate: item_id should be in the known set
            if item_id in item_order or item_id in _ITEM_TITLES:
                candidates.append((i, item_id, rest))

    if not candidates:
        # No Item headers found — try title-only matching as last resort
        return _segment_by_titles(lines, item_order)

    # ── Pass 2: detect TOC clusters ─────────────────────────────────
    # A TOC cluster is a group of ≥6 Item headers within a 50-line window.
    # This works regardless of document position (JPM has TOC at ~80%).
    toc_lines: set[int] = set()
    cand_positions = [c[0] for c in candidates]
    for i, pos in enumerate(cand_positions):
        # Count headers within ±40 lines
        nearby = sum(1 for p in cand_positions if abs(p - pos) <= 40 and p != pos)
        if nearby >= 5:
            toc_lines.add(pos)

    # Also mark early-doc headers as TOC
    early_boundary = int(n_lines * 0.12)

    # ── Pass 3: score each candidate ──────────────────────────────────
    scored: list[tuple[int, str, str, float]] = []
    for idx, (line_idx, item_id, rest) in enumerate(candidates):
        score = 0.0
        full_line = lines[line_idx].strip()

        # Strong penalty: in a detected TOC cluster
        if line_idx in toc_lines:
            score -= 5.0
        # Penalty: in the very early part of the document (cover page / initial TOC)
        if line_idx < early_boundary:
            score -= 2.0
        # Penalty: trailing page number (e.g., "Risk Factors. 9-31")
        if re.search(r"(?:\b\d{1,4}(?:\s*[-–]\s*\d{1,4})?)\s*$", rest):
            score -= 3.0

        # Big penalty: no title text at all — just "Item 7" with nothing after.
        # These are page footers/headers repeated throughout the document.
        if not rest or len(rest.strip()) < 3:
            score -= 4.0
            # Extra penalty if near a standalone page number or "PART X"
            for j in range(max(0, line_idx - 3), min(n_lines, line_idx + 3)):
                near = lines[j].strip()
                if re.match(r"^\d{1,4}$", near) or re.match(r"^PART\s+[IVX]+", near, re.I):
                    score -= 2.0
                    break

        # Bonus: has substantive title text matching the expected title.
        # Handle split titles: HTML-to-text often breaks words across lines.
        # If rest is a short fragment (1-5 chars), it's likely a broken word;
        # concatenate WITHOUT space to rejoin "B"+"USINESS" → "BUSINESS".
        title_pat = _ITEM_TITLES.get(item_id)
        combined_rest = rest
        if len(rest) < 60:
            parts = [rest]
            for k in range(1, 4):
                if line_idx + k >= n_lines:
                    break
                nxt = lines[line_idx + k].strip()
                if not nxt:
                    continue
                if _ITEM_HDR_RE.match(nxt):
                    break
                # If last part is very short, likely a broken word — no space
                if len(parts[-1]) < 6:
                    parts[-1] = parts[-1] + nxt
                else:
                    parts.append(nxt)
            combined_rest = " ".join(parts)
        # Normalize: collapse whitespace
        combined_norm = re.sub(r"\s+", " ", combined_rest).strip()
        if title_pat and title_pat.search(combined_norm):
            score += 3.0
        elif rest and len(rest.strip()) > 10:
            score += 1.0  # has some title text even if not matching pattern

        # Bonus: followed by substantive content (next non-empty line is long)
        for j in range(line_idx + 1, min(line_idx + 6, n_lines)):
            nxt = lines[j].strip()
            if nxt:
                if len(nxt) > 120:
                    score += 3.0
                elif len(nxt) > 60 and not _ITEM_HDR_RE.match(nxt):
                    score += 1.5
                elif _ITEM_HDR_RE.match(nxt):
                    score -= 2.0  # another header right after = likely TOC
                break

        scored.append((line_idx, item_id, rest, score))

    # ── Pass 3: pick best candidate per item ──────────────────────────
    by_item: dict[str, list[tuple[int, float]]] = {}
    for line_idx, item_id, _rest, score in scored:
        by_item.setdefault(item_id, []).append((line_idx, score))

    best: dict[str, int] = {}  # item_id → line_idx
    for item_id, entries in by_item.items():
        # Prefer highest-scoring candidate
        entries.sort(key=lambda e: (-e[1], e[0]))
        best_entry = entries[0]
        # If the best score is very negative but there's a non-TOC one, prefer it
        if best_entry[1] < 0:
            for li, sc in entries:
                if li not in toc_lines and li >= early_boundary and sc > best_entry[1] - 1:
                    best_entry = (li, sc)
                    break
        best[item_id] = best_entry[0]

    # ── Pass 4: enforce ordering + build ranges ───────────────────────
    # Items must appear in document order.  If two items are out-of-order
    # relative to the canonical 10-K sequence, drop the worse-scoring one.
    ordered = sorted(best.items(), key=lambda kv: kv[1])

    # Validate ordering matches canonical order
    final_items = _enforce_order(ordered, item_order)

    # Build (start, end) ranges
    section_ranges: dict[str, tuple[int, int]] = {}
    sorted_final = sorted(final_items.items(), key=lambda kv: kv[1])
    for idx, (item_id, start_line) in enumerate(sorted_final):
        if idx + 1 < len(sorted_final):
            end_line = sorted_final[idx + 1][1]
        else:
            # Last section runs to end of document (minus signatures)
            end_line = _find_signatures_start(lines, start_line)
        section_ranges[item_id] = (start_line, end_line)

    log.info(
        "Segmented filing into %d sections: %s",
        len(section_ranges),
        ", ".join(f"Item {k}" for k in sorted(section_ranges, key=_item_sort_key)),
    )
    return SectionMap(lines, section_ranges)


# ═══════════════════════════════════════════════════════════════════════════
#  LLM fallback — Line-ID-Based (LIB) prompting
# ═══════════════════════════════════════════════════════════════════════════

_LIB_SYSTEM_PROMPT = """\
You are a financial document analyst. Your task is to identify the starting \
lines of each Item in a 10-K annual report. Each line has a numeric line-ID \
followed by its content (first 20 words only).

Rules:
- Skip Table-of-Contents references. Only identify lines where the actual \
  section **content** begins (not TOC page references).
- If an item is not present, output NA.
- Output ONLY in this format, one per line:
  Item 1,<line_id>
  Item 1A,<line_id>
  ...etc
"""

_LIB_FEW_SHOT = """\
Example:
=====
0 Table of Contents
1 UNITED STATES SECURITIES AND EXCHANGE COMMISSION
5 FORM 10-K
10 ANNUAL REPORT PURSUANT TO SECTION 13
30 TABLE OF CONTENTS
31 Item 1 Business 5
32 Item 1A Risk Factors 20
33 Item 7 MD&A 45
50 PART I
51 ITEM 1. BUSINESS
52 Overview We are a leading provider of technology solutions
53 serving customers in over 50 countries worldwide
150 ITEM 1A. RISK FACTORS
151 Investing in our securities involves risk.
400 ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
401 The following discussion should be read in conjunction
=====
Output:
Item 1,51
Item 1A,150
Item 7,400
"""


def llm_segment_filing(
    text: str,
    form_type: str = "10-K",
) -> dict[str, int] | None:
    """Use Claude with LIB prompting to identify section boundaries.

    Returns a dict of item_id → start_line_index, or None on failure.
    """
    try:
        from sec_mcp.config import get_config
        cfg = get_config()
        api_key = cfg.anthropic_api_key
        if not api_key:
            return None
    except Exception:
        return None

    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed — skipping LLM segmentation")
        return None

    lines = text.split("\n")

    # Build condensed input: line-ID + first L words
    L = 20
    condensed: list[str] = []
    for i, line in enumerate(lines):
        words = line.split()[:L]
        if words:
            condensed.append(f"{i} {' '.join(words)}")

    # Truncate to fit context window (~3500 lines max)
    if len(condensed) > 3500:
        # Keep first 500 + last 3000 (actual content is in the back)
        condensed = condensed[:500] + ["... (truncated) ..."] + condensed[-3000:]

    items_list = ", ".join(f"Item {x}" for x in _10K_ITEM_ORDER[:18])
    user_msg = (
        f"{_LIB_FEW_SHOT}\n"
        f"Now identify items in this 10-K report.\n"
        f"Items to find: {items_list}\n"
        f"=====\n"
        + "\n".join(condensed)
        + "\n=====\nOutput:"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=_LIB_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        reply = resp.content[0].text.strip()
    except Exception as exc:
        log.warning("LLM segmentation call failed: %s", exc)
        return None

    # Parse response
    result: dict[str, int] = {}
    for line in reply.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",", 1)
        item_label = parts[0].strip()
        line_id_str = parts[1].strip()
        if line_id_str.upper() == "NA":
            continue
        # Extract item number from "Item 7A" → "7A"
        m = re.match(r"Item\s+(\d+[A-Za-z]?)", item_label, re.I)
        if not m:
            continue
        item_id = m.group(1).upper()
        try:
            line_id = int(line_id_str)
            if 0 <= line_id < len(lines):
                result[item_id] = line_id
        except ValueError:
            continue

    if result:
        log.info("LLM segmentation found %d items: %s", len(result), list(result.keys()))
    return result if result else None


# ═══════════════════════════════════════════════════════════════════════════
#  Cache
# ═══════════════════════════════════════════════════════════════════════════

class _LRUCache(OrderedDict):
    """Simple LRU cache with a max-size limit."""

    def __init__(self, maxsize: int = 64):
        super().__init__()
        self._maxsize = maxsize

    def get_or_none(self, key: str) -> SectionMap | None:
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key: str, value: SectionMap) -> None:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self._maxsize:
            self.popitem(last=False)


_cache = _LRUCache(maxsize=64)


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def extract_section(
    text: str,
    section: str,
    *,
    accession: str = "",
    form_type: str = "10-K",
    raw_html: str = "",
) -> str | None:
    """Extract a specific section from filing plain text.

    This is the main entry point.  It builds (or retrieves from cache)
    a full section map, then returns the requested section.

    Args:
        text:       Filing plain text (HTML already stripped).
        section:    Friendly name ("mda", "risk_factors") or item id ("7").
        accession:  Filing accession number (for caching).
        form_type:  "10-K" or "10-Q".
        raw_html:   Original HTML (used for HTML-aware fallback).

    Returns:
        Section text if found (≥200 chars), else None.
    """
    # Resolve section name → item id
    item_id = SECTION_ALIASES.get(section.lower().strip())
    if item_id is None:
        # Maybe it's already an item id like "7" or "Item 7"
        m = re.match(r"(?:Item\s+)?(\d+[A-Za-z]?)", section, re.I)
        item_id = m.group(1).upper() if m else section.upper()

    # Check cache
    cache_key = accession or str(hash(text[:5000]))
    smap = _cache.get_or_none(cache_key)

    if smap is None:
        # Build section map
        smap = segment_filing(text, form_type)

        # If the requested item wasn't found, try LLM fallback
        if item_id not in smap.sections:
            llm_result = llm_segment_filing(text, form_type)
            if llm_result:
                # Merge LLM results into the section map
                smap = _merge_llm_results(smap, llm_result)

        _cache.put(cache_key, smap)

    result = smap.get_section_text(item_id)

    # Validate result quality
    if result and len(result.strip()) >= 200:
        # Reject if it looks like a signature block
        if _is_signature_block(result):
            log.warning("Rejected signature block for Item %s", item_id)
            return None
        return result

    # Last-resort: try HTML-aware extraction if we have raw HTML
    if raw_html:
        html_result = _extract_from_html(raw_html, item_id)
        if html_result and len(html_result.strip()) >= 200 and not _is_signature_block(html_result):
            return html_result

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _is_reference_stub(text: str) -> bool:
    """Return True if text is a short stub that references content elsewhere."""
    lower = text.lower()
    ref_phrases = [
        "incorporated herein by reference",
        "incorporated by reference",
        "is on pages",
        "are on pages",
        "appears on pages",
        "appear on pages",
        "see pages",
        "refer to",
        "included elsewhere in this",
        "set forth in",
        "should be read in conjunction with",
    ]
    return any(p in lower for p in ref_phrases)


def _find_actual_content(
    lines: list[str], item_id: str,
) -> str | None:
    """Search the full document for actual section content when
    the formal Item header is just a stub/reference.

    Common for banks (JPM, GS, BAC) where MD&A is in the annual report
    section that precedes the 10-K wrapper items.
    """
    title_pat = _ITEM_TITLES.get(item_id)
    if not title_pat:
        return None

    n_lines = len(lines)

    # Find ALL candidate heading lines that match the section title
    heading_candidates: list[tuple[int, int]] = []  # (line_idx, content_chars_after)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue
        if not title_pat.search(stripped):
            continue
        # Must look like a heading: short line, not embedded in a paragraph
        if len(stripped) > 150:
            continue

        # Measure substantive content in the next 300 lines
        content_chars = 0
        content_lines = 0
        for j in range(i + 1, min(i + 300, n_lines)):
            nxt = lines[j].strip()
            if nxt:
                content_chars += len(nxt)
                content_lines += 1
            # Stop counting if we hit another major Item header
            if _ITEM_HDR_RE.match(nxt):
                break
        heading_candidates.append((i, content_chars))

    if not heading_candidates:
        return None

    # Pick the heading with the most content after it
    heading_candidates.sort(key=lambda x: -x[1])
    best_start, best_content = heading_candidates[0]

    if best_content < 500:
        return None

    # Find a natural end boundary
    end = n_lines
    content_started = False
    for j in range(best_start + 1, n_lines):
        stripped = lines[j].strip()
        if stripped:
            content_started = True
        # Stop at a formal Item header
        if content_started and _ITEM_HDR_RE.match(stripped):
            end = j
            break
    # Cap at a reasonable length (500 lines)
    end = min(end, best_start + 500)

    raw = "\n".join(lines[best_start:end])
    return _clean_section_text(raw)


def _item_sort_key(item_id: str) -> tuple[int, str]:
    """Sort key for item ids: numeric part first, then alpha suffix."""
    m = re.match(r"(\d+)(.*)", item_id)
    if m:
        return (int(m.group(1)), m.group(2))
    return (999, item_id)


def _enforce_order(
    ordered: list[tuple[str, int]],
    canonical: list[str],
) -> dict[str, int]:
    """Remove items that violate the canonical ordering.

    Greedily keeps items that appear in document order consistent
    with the canonical 10-K item sequence.
    """
    canon_idx = {item: i for i, item in enumerate(canonical)}
    result: dict[str, int] = {}
    last_canon = -1
    for item_id, line_idx in ordered:
        ci = canon_idx.get(item_id, -1)
        if ci > last_canon:
            result[item_id] = line_idx
            last_canon = ci
        else:
            # Out of order — skip (likely a stale TOC reference)
            log.debug("Dropping out-of-order Item %s at line %d", item_id, line_idx)
    return result


def _find_signatures_start(lines: list[str], after_line: int) -> int:
    """Find where signatures / Power of Attorney begin after a given line."""
    sig_patterns = [
        re.compile(r"power\s+of\s+attorney", re.I),
        re.compile(r"know\s+all\s+persons?\s+by\s+these\s+presents", re.I),
        re.compile(
            r"pursuant\s+to\s+the\s+requirements\s+of\s+the\s+securities\s+exchange\s+act"
            r"\s+of\s+1934.*?this\s+report\s+has\s+been\s+signed",
            re.I,
        ),
    ]
    # Also detect dense /s/ blocks
    sig_run = 0
    for i in range(after_line, len(lines)):
        line = lines[i].strip()
        for pat in sig_patterns:
            if pat.search(line):
                return i
        if re.search(r"/s/", line):
            sig_run += 1
            if sig_run >= 3:
                return i - 2
        else:
            sig_run = 0
    return len(lines)


def _segment_by_titles(lines: list[str], item_order: list[str]) -> SectionMap:
    """Fallback segmenter when no 'Item N' headers are found.

    Looks for standalone title lines like 'RISK FACTORS' or
    'MANAGEMENT'S DISCUSSION AND ANALYSIS'.
    """
    n_lines = len(lines)
    toc_boundary = int(n_lines * 0.12)
    found: dict[str, int] = {}

    for i, line in enumerate(lines):
        if i < toc_boundary:
            continue
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue
        for item_id, pat in _ITEM_TITLES.items():
            if item_id in found:
                continue
            if pat.search(stripped):
                # Verify it looks like a heading (short, possibly all-caps)
                if len(stripped) < 120:
                    found[item_id] = i

    ordered = sorted(found.items(), key=lambda kv: kv[1])
    final = _enforce_order(ordered, item_order)

    section_ranges: dict[str, tuple[int, int]] = {}
    sorted_final = sorted(final.items(), key=lambda kv: kv[1])
    for idx, (item_id, start_line) in enumerate(sorted_final):
        if idx + 1 < len(sorted_final):
            end_line = sorted_final[idx + 1][1]
        else:
            end_line = _find_signatures_start(lines, start_line)
        section_ranges[item_id] = (start_line, end_line)

    return SectionMap(lines, section_ranges)


def _merge_llm_results(smap: SectionMap, llm: dict[str, int]) -> SectionMap:
    """Merge LLM-detected boundaries into an existing SectionMap."""
    merged = dict(smap.sections)
    all_starts: dict[str, int] = {}
    for item_id, (s, _e) in merged.items():
        all_starts[item_id] = s
    for item_id, start in llm.items():
        if item_id not in all_starts:
            all_starts[item_id] = start

    # Rebuild ranges
    ordered = sorted(all_starts.items(), key=lambda kv: kv[1])
    new_ranges: dict[str, tuple[int, int]] = {}
    for idx, (item_id, start_line) in enumerate(ordered):
        if idx + 1 < len(ordered):
            end_line = ordered[idx + 1][1]
        else:
            end_line = _find_signatures_start(smap.lines, start_line)
        new_ranges[item_id] = (start_line, end_line)

    return SectionMap(smap.lines, new_ranges)


def _clean_section_text(text: str) -> str:
    """Clean up extracted section text."""
    # Remove excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Truncate at signature blocks
    for pat in [
        r"\n\s*Power\s+of\s+Attorney\s*\n",
        r"\n\s*KNOW\s+ALL\s+PERSONS\s+BY\s+THESE\s+PRESENTS",
        r"Pursuant\s+to\s+the\s+requirements\s+of\s+the\s+Securities\s+Exchange"
        r"\s+Act\s+of\s+1934.*?this\s+report\s+has\s+been\s+signed",
    ]:
        m = re.search(pat, text, re.I | re.DOTALL)
        if m:
            text = text[:m.start()].strip()
    return text


def _is_signature_block(text: str) -> bool:
    """Return True if text looks like a signature / Power of Attorney block."""
    if not text or len(text.strip()) < 100:
        return False
    head = text[:3000].lower()
    markers = [
        "power of attorney",
        "know all persons by these presents",
        "pursuant to the requirements of the securities exchange act of 1934, this report has been signed",
    ]
    if any(m in head for m in markers):
        return True
    sig_count = len(re.findall(r"/s/\s*[A-Z][a-z]+", text[:5000]))
    return sig_count >= 3


def _extract_from_html(html: str, item_id: str) -> str | None:
    """HTML-aware fallback using BeautifulSoup anchors/bookmarks."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    item_label = f"Item {item_id}"
    item_num_clean = item_id.replace(" ", "").lower()

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None

    # Find anchor tags or heading elements that reference this item
    anchors: list[Any] = []
    for tag in soup.find_all(
        ["a", "div", "span", "p", "h1", "h2", "h3", "h4", "b", "font"]
    ):
        tag_name_attr = (tag.get("name") or "").lower().replace(" ", "").replace("_", "")
        tag_id_attr = (tag.get("id") or "").lower().replace(" ", "").replace("_", "")
        tag_text = tag.get_text(strip=True)

        # Match by id/name attribute
        if item_num_clean in tag_name_attr or item_num_clean in tag_id_attr:
            anchors.append(tag)
            continue

        # Match by text content (short elements only)
        if len(tag_text) < 120 and item_label.lower() in tag_text.lower():
            anchors.append(tag)

    if not anchors:
        return None

    # Prefer anchors in the latter part of the document (skip TOC)
    best = anchors[-1]

    # Build the next-item stopping pattern
    next_item_re = re.compile(r"Item\s+\d+[A-Za-z]*[\.\s\:\-]", re.I)

    parts: list[str] = []
    total_len = 0
    for string_node in best.find_all_next(string=True):
        txt = string_node.strip()
        if not txt:
            continue
        # Stop at next Item header (different item)
        if total_len > 500 and next_item_re.search(txt):
            if item_label.lower() not in txt.lower()[:30]:
                break
        parts.append(txt)
        total_len += len(txt)
        if total_len > 150_000:
            break

    return "\n".join(parts) if parts else None
