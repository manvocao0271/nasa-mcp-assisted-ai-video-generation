"""Shared utility helpers used by multiple page files."""
from __future__ import annotations

import httpx


def fetch_thumb(url: str) -> bytes | str:
    """Return image bytes for NASA Image Library URLs (proxy to bypass 403).

    NASA Image Library assets (images-assets.nasa.gov) return 403/timeout when
    loaded directly by the browser.  We fetch them server-side and hand Streamlit
    raw bytes instead, which always works.  We also swap ~large / ~orig variants
    for ~thumb to keep previews fast.
    """
    _PROXY_DOMAINS = ("images-assets.nasa.gov",)
    if not any(d in url for d in _PROXY_DOMAINS):
        return url  # fast path — most URLs load fine in the browser
    thumb_url = (
        url.replace("~large.", "~thumb.")
        .replace("~orig.", "~thumb.")
        .replace("~medium.", "~thumb.")
    )
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            r = client.get(thumb_url)
        if r.status_code < 400:
            return r.content
        r = client.get(url)
        if r.status_code < 400:
            return r.content
    except Exception:
        pass
    return url  # fall back to URL; Streamlit will show broken-image icon
