from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup, UnicodeDammit
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from readability import Document


REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class ExtractRequest(BaseModel):
    url: HttpUrl


app = FastAPI(title="Static Web Extractor", version="0.1.0")


def _clean_text(html_fragment: str) -> str:
    soup = BeautifulSoup(html_fragment, "lxml")
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


def extract_page_content(url: str) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    decoded_html = UnicodeDammit(response.content, is_html=True).unicode_markup
    if decoded_html is None:
        raise ValueError("Failed to decode page content")

    doc = Document(decoded_html)
    title = doc.short_title()
    content_html = doc.summary(html_partial=True)
    content_text = _clean_text(content_html)

    return {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "title": title,
        "text": content_text,
        "text_length": len(content_text),
    }


@app.post("/extract")
def extract(payload: ExtractRequest) -> dict[str, Any]:
    try:
        return extract_page_content(str(payload.url))
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
