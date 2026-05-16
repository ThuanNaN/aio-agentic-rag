"""Strip HTML and normalize Vietnamese legal text."""
from __future__ import annotations
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from tqdm import tqdm


def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Preserve block-level breaks as newlines
    for tag in soup.find_all(["p", "div", "br", "li", "tr"]):
        tag.insert_before("\n")
    return soup.get_text(separator=" ")


def _normalize(text: str) -> str:
    # NFC unicode normalization (important for Vietnamese diacritics)
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace while preserving single newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document(doc: Document) -> Document:
    """Return a new Document with cleaned plain-text page_content."""
    raw = doc.page_content
    if "<" in raw and ">" in raw:
        raw = _strip_html(raw)
    cleaned = _normalize(raw)
    return Document(page_content=cleaned, metadata=doc.metadata)


def clean_documents(docs: list[Document], workers: int = 1) -> list[Document]:
    if workers <= 1:
        return [clean_document(d) for d in tqdm(docs, desc="Cleaning documents", unit="doc")]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(tqdm(
            executor.map(clean_document, docs, chunksize=64),
            total=len(docs),
            desc="Cleaning documents",
            unit="doc",
        ))
