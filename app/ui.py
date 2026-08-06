"""Single-page web UI for the Quran assistant.

Served at ``GET /`` by ``app.main`` via the ``INDEX_HTML`` string. The markup lives in the sibling
``index.html`` file (authored as real HTML, not a Python-escaped string, so the CSS/JS is
maintainable and free of escaping hazards); it is read once at import and cached. The page is
fully self-contained at runtime — inline CSS + JS, no build step, same-origin fetches to the
``/v1/*`` API. It POSTs to ``/v1/ask/stream`` (grounded streaming) with graceful fallback to
``/v1/ask``, renders cited evidence verses, and keeps conversation history via the session API.
"""

from __future__ import annotations

from pathlib import Path

_INDEX_PATH = Path(__file__).with_name("index.html")

# Minimal fallback so the server still starts and explains itself if index.html is missing.
_FALLBACK_HTML = (
    "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<title>Quran AI Assistant</title></head><body style=\"font-family:sans-serif;"
    "max-width:640px;margin:60px auto;padding:0 20px;color:#17201b\">"
    "<h1>Quran AI Assistant</h1>"
    "<p>The UI template <code>app/index.html</code> could not be found, so the interface "
    "cannot be rendered. The JSON API is still available at <code>/v1/ask</code>, "
    "<code>/v1/search</code>, and <code>/v1/health</code>.</p>"
    "</body></html>"
)


def _load_index_html() -> str:
    try:
        return _INDEX_PATH.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_HTML


INDEX_HTML = _load_index_html()
