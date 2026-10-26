"""Deterministic text cleanup that preserves financial symbols and numbers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from finomena.data.sources import Document

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    printable = "".join(
        char for char in normalized if char == "\n" or char == "\t" or char.isprintable()
    )
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in printable.split("\n")]
    return _EXCESS_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


@dataclass(frozen=True)
class CleaningReport:
    documents_read: int
    documents_retained: int
    empty_removed: int
    duplicates_removed: int
    character_count: int


def clean_documents(documents: list[Document]) -> tuple[list[Document], CleaningReport]:
    cleaned: list[Document] = []
    seen: set[str] = set()
    empty_removed = 0
    duplicates_removed = 0
    for document in documents:
        text = clean_text(document.text)
        if not text:
            empty_removed += 1
            continue
        if text in seen:
            duplicates_removed += 1
            continue
        seen.add(text)
        cleaned.append(
            Document(
                title=clean_text(document.title),
                text=text,
                url=document.url,
                published_at=document.published_at,
                source=document.source,
            )
        )
    report = CleaningReport(
        documents_read=len(documents),
        documents_retained=len(cleaned),
        empty_removed=empty_removed,
        duplicates_removed=duplicates_removed,
        character_count=sum(len(document.text) for document in cleaned),
    )
    return cleaned, report
