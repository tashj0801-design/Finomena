from finomena.data.sources import FEED_URL, fetch_press_releases


def test_source_fetch_filters_to_press_release_pages(monkeypatch) -> None:
    feed = b"""<?xml version="1.0"?>
    <rss><channel>
      <item><title>Release</title><link>https://www.ecb.europa.eu/press/pr/x.en.html</link>
        <pubDate>Mon, 26 Oct 2026 10:00:00 +0100</pubDate></item>
      <item><title>Speech</title><link>https://www.ecb.europa.eu/press/key/x.en.html</link>
        <pubDate>Mon, 26 Oct 2026 10:00:00 +0100</pubDate></item>
    </channel></rss>"""
    page = b"<html><nav>Menu</nav><main><article><p>Revenue rose $2.4 and 3%.</p></article></main></html>"
    requested: list[str] = []

    def fake_fetch(url: str) -> bytes:
        requested.append(url)
        return feed if url == FEED_URL else page

    monkeypatch.setattr("finomena.data.sources._fetch", fake_fetch)
    documents = fetch_press_releases(max_documents=10, delay_seconds=0)
    assert len(documents) == 1
    assert len(requested) == 2
    assert "Revenue rose $2.4 and 3%." in documents[0].text
    assert "Menu" not in documents[0].text
    assert documents[0].published_at.startswith("2026-10-26")
