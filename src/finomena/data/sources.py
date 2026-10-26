"""On-demand acquisition of attributable ECB English press releases."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

FEED_URL = "https://www.ecb.europa.eu/rss/press.html"
SOURCE_NAME = "European Central Bank (ECB) press releases"
USER_AGENT = "Finomena educational language-model project/0.1 (research; respectful rate limit)"


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    url: str
    published_at: str
    source: str = SOURCE_NAME


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._main_depth = 0
        self._saw_main = False
        self._parts: list[str] = []
        self._fallback_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "nav", "footer", "header"}:
            self._skip_depth += 1
        if tag in {"main", "article"}:
            self._main_depth += 1
            self._saw_main = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "nav", "footer", "header"}:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in {"main", "article"}:
            self._main_depth = max(0, self._main_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._fallback_parts.append(text)
                if self._main_depth:
                    self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts if self._saw_main and self._parts else self._fallback_parts)


def _fetch(url: str, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _valid_feed_date(value: str) -> str:
    try:
        date = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    return date.isoformat() if isinstance(date, datetime) else ""


def fetch_press_releases(max_documents: int = 40, delay_seconds: float = 1.0) -> list[Document]:
    """Fetch current feed items and their pages, filtering out speeches/interviews."""
    if max_documents < 1:
        raise ValueError("max_documents must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    feed = ET.fromstring(_fetch(FEED_URL))
    items: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()
    for item in feed.findall("./channel/item"):
        link = (item.findtext("link") or "").strip().replace("https://www.ecb.europa.eu//", "https://www.ecb.europa.eu/")
        title = (item.findtext("title") or "").strip()
        published_at = _valid_feed_date((item.findtext("pubDate") or "").strip())
        if "/press/pr/" not in link or not link.endswith(".html") or link in seen_urls:
            continue
        seen_urls.add(link)
        items.append((title, link, published_at))
        if len(items) >= max_documents:
            break

    documents: list[Document] = []
    for index, (title, link, published_at) in enumerate(items):
        if index and delay_seconds:
            time.sleep(delay_seconds)
        parser = _VisibleTextParser()
        parser.feed(_fetch(link).decode("utf-8", errors="replace"))
        text = parser.text()
        if text:
            documents.append(
                Document(title=title, text=text, url=link, published_at=published_at)
            )
    return documents


def write_jsonl(documents: list[Document], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for document in documents:
            file.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"document at line {line_number} has no text string")
            documents.append(
                Document(
                    title=str(row.get("title", "")),
                    text=row["text"],
                    url=str(row.get("url", "")),
                    published_at=str(row.get("published_at", "")),
                    source=str(row.get("source", SOURCE_NAME)),
                )
            )
    return documents
