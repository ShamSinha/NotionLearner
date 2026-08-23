from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse


VIEWER_SCHEMES = {"chrome-extension", "moz-extension", "extension"}


def _decoded(value: str) -> str:
    result = value or ""
    for _ in range(3):
        decoded = unquote(result)
        if decoded == result:
            break
        result = decoded
    return result


def unwrap_viewer_url(value: object) -> str:
    """Recover an HTTP(S) source hidden inside a browser PDF-viewer URL."""
    raw = str(value or "").strip()
    if raw.lower().startswith(("http://", "https://")):
        return raw

    parsed = urlparse(raw)
    if parsed.scheme.lower() not in VIEWER_SCHEMES:
        return raw

    query = parse_qs(parsed.query)
    candidates = [
        *(query.get("file") or []),
        *(query.get("url") or []),
        *(query.get("src") or []),
        parsed.path.lstrip("/"),
        parsed.fragment,
    ]
    for candidate in candidates:
        decoded = _decoded(candidate).strip()
        http_at = min(
            (index for index in (decoded.find("https://"), decoded.find("http://")) if index >= 0),
            default=-1,
        )
        if http_at >= 0:
            return decoded[http_at:]
    return raw
