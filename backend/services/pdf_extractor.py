from __future__ import annotations

import io
from typing import Optional
from urllib.parse import urlparse

import httpx


def is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or "arxiv.org/pdf/" in url.lower()


def extract_pdf_text(url: str = "", pdf_bytes: Optional[bytes] = None, max_chars: int = 80000) -> str:
    """Extract text from a PDF URL or raw bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ValueError("PyMuPDF not installed — pip install pymupdf") from exc

    data = pdf_bytes
    if data is None:
        if not url:
            raise ValueError("No PDF url/bytes provided")
        # arxiv abs → pdf
        if "arxiv.org/abs/" in url:
            url = url.replace("/abs/", "/pdf/") + ".pdf"
            if url.endswith(".pdf.pdf"):
                url = url[:-4]
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content

    doc = fitz.open(stream=data, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
        if sum(len(p) for p in parts) >= max_chars:
            break
    doc.close()
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("PDF contained no extractable text")
    return text[:max_chars]
