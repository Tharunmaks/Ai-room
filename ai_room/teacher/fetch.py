"""Fetch open textbooks from Wikipedia, Simple English Wikipedia and Wikibooks.

Wikimedia asks API clients to identify themselves and to go slowly; this fetcher
sends a descriptive User-Agent, waits between requests and honours Retry-After.
Text is available under CC BY-SA 4.0; every passage keeps its source URL so the
teacher can cite it.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable
from urllib.parse import quote

import httpx

from ..curriculum import SUBJECTS, topic_queries
from .library import Library

USER_AGENT = "AI-Room/0.2 (+https://github.com/tharunmaks/ai-room; educational use; polite crawler)"
SITES = {
    "simple": ("simple.wikipedia.org", "simple", "Simple English Wikipedia"),
    "wikipedia": ("en.wikipedia.org", "standard", "Wikipedia"),
    "wikibooks": ("en.wikibooks.org", "standard", "Wikibooks"),
}
LICENSE = "CC BY-SA 4.0"


class FetchError(RuntimeError):
    pass


class Wikimedia:
    def __init__(self, client: httpx.Client | None = None, min_interval: float = 0.8, sleep: Callable[[float], None] = time.sleep):
        self.client = client or httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, follow_redirects=True)
        self.min_interval = min_interval
        self.sleep = sleep
        self._last = 0.0

    def _get(self, host: str, params: dict) -> dict:
        url = f"https://{host}/w/api.php"
        params = {**params, "format": "json", "formatversion": "2", "utf8": "1"}
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.client.get(url, params=params)
            except httpx.HTTPError as e:
                if attempt == 4:
                    raise FetchError(f"Network error talking to {host}: {e}") from e
                self.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 503):
                retry = r.headers.get("retry-after", "")
                self.sleep(min(float(retry) if retry.isdigit() else 5 * (attempt + 1), 30))
                continue
            if r.status_code in (401, 403):
                raise FetchError(f"{host} refused the request (HTTP {r.status_code}). Wikimedia blocks connections that crawl too fast or share a busy IP — "
                                 "wait a while and retry, or import your own textbooks instead.")
            if r.status_code >= 400:
                raise FetchError(f"{host} answered HTTP {r.status_code}")
            try:
                data = r.json()
            except ValueError as e:
                raise FetchError(f"{host} returned something that is not JSON") from e
            if "error" in data:
                raise FetchError(f"{host}: {data['error'].get('info', data['error'])}")
            return data
        raise FetchError(f"{host} is rate-limiting this connection; try again later")

    def search(self, host: str, query: str, limit: int = 3) -> list[str]:
        data = self._get(host, {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "srnamespace": 0})
        return [h["title"] for h in data.get("query", {}).get("search", [])]

    def extract(self, host: str, title: str) -> str | None:
        data = self._get(host, {"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki", "redirects": 1, "titles": title})
        pages = data.get("query", {}).get("pages", [])
        pages = list(pages.values()) if isinstance(pages, dict) else pages  # formatversion 1 vs 2
        if not pages or "missing" in pages[0] or "invalid" in pages[0]:
            return None
        text = (pages[0].get("extract") or "").strip()
        if len(text) < 600 or "may refer to" in text[:400] or text.lower().startswith(("list of", "this is a list")):
            return None
        return re.sub(r"\n{3,}", "\n\n", text)


def page_url(host: str, title: str) -> str:
    return f"https://{host}/wiki/{quote(title.replace(' ', '_'))}"


def fetch_subject(lib: Library, subject: str, *, wiki: Wikimedia | None = None, sources: tuple[str, ...] = ("simple", "wikipedia", "wikibooks"),
                  per_site: int = 2, progress: Callable[[str, int, int], None] | None = None, stop: threading.Event | None = None) -> dict:
    """Fetch textbook text for every topic in a subject's 12-class syllabus."""
    wiki = wiki or Wikimedia()
    queries = topic_queries(subject)
    name = SUBJECTS[subject].name
    total = len(queries)
    summary = {"subject": subject, "added": 0, "skipped": 0, "errors": [], "topics": total}
    for i, (phrase, _grade) in enumerate(queries, 1):
        if stop and stop.is_set():
            summary["stopped"] = True
            break
        query = phrase if name.lower() in phrase.lower() else f"{phrase} {name}"
        if progress:
            progress(f"Reading about “{phrase}”", i - 1, total)
        for key in sources:
            host, level, label = SITES[key]
            try:
                titles = wiki.search(host, query, limit=per_site + 1)
                taken = 0
                for title in titles:
                    if taken >= per_site:
                        break
                    if lib.has_document(subject, title, label):
                        summary["skipped"] += 1
                        taken += 1
                        continue
                    text = wiki.extract(host, title)
                    if not text:
                        continue
                    doc = lib.add_document(subject, title, text, source=label, url=page_url(host, title), level=level, topic=phrase, license=LICENSE)
                    if doc:
                        summary["added"] += 1
                        taken += 1
            except FetchError as e:
                summary["errors"].append(str(e))
                if len(summary["errors"]) >= 6:
                    if progress:
                        progress("Too many errors — stopping.", i, total)
                    return summary
    if progress:
        progress("Done", total, total)
    return summary
