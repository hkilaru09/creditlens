"""Fetch a filing, strip it to plain text, chunk, embed, and store it."""
from __future__ import annotations

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ..edgar.client import EdgarClient
from .chunker import chunk_text
from .embeddings import embed_texts
from .vectorstore import FilingVectorStore

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def index_filing(
    store: FilingVectorStore,
    edgar: EdgarClient,
    ticker: str,
    cik: str,
    filing: dict,
) -> int:
    html = edgar.fetch_filing_html(cik, filing["accessionNumber"], filing["primaryDocument"])
    text = html_to_text(html)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embed_texts([c["text"] for c in chunks])
    ids = [f"{ticker}-{filing['accessionNumber']}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "ticker": ticker,
            "form": filing["form"],
            "filingDate": filing["filingDate"],
            "accessionNumber": filing["accessionNumber"],
            "start": c["start"],
            "end": c["end"],
        }
        for c in chunks
    ]
    store.add(ids=ids, embeddings=embeddings, documents=[c["text"] for c in chunks], metadatas=metadatas)
    return len(chunks)
