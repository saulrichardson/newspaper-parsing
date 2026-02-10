from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable

TEXT_KEYS = {
    "text",
    "article",
    "paragraph",
    "content",
    "reference",
    "number",
    "footnote",
    "caption",
    "image_caption",
    "table_footnote",
    "image_footnote",
    "table_caption",
}
TITLE_KEYS = {
    "title",
    "header",
    "headline",
    "masthead",
    "author",
    "doc_title",
    "page_number",
    "paragraph_title",
    "newspaper_header",
}
TABLE_KEYS = {"table"}
IMAGE_KEYS = {"image", "figure", "photo", "advert", "cartoon", "photograph", "logo", "picture"}


def normalize_label(raw: str) -> str:
    x = (raw or "").strip().lower()
    if any(k in x for k in TEXT_KEYS):
        return "text"
    if any(k in x for k in TITLE_KEYS):
        return "title"
    if any(k in x for k in TABLE_KEYS):
        return "table"
    if any(k in x for k in IMAGE_KEYS):
        return "image"
    return "other"


def label_counts(boxes: Iterable[Dict[str, Any]], key: str = "source_label") -> Dict[str, int]:
    c = Counter()
    for b in boxes:
        c[str(b.get(key, ""))] += 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
