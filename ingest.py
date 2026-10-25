"""
ingest.py -- build the vector index for the RAG demo.

Pipeline: load .txt documents from data/ -> split into overlapping chunks ->
embed each chunk locally with sentence-transformers (all-MiniLM-L6-v2) ->
build a FAISS inner-product index over L2-normalized vectors (so inner
product == cosine similarity) -> save index + chunk metadata to index/.

Run:  python ingest.py
Re-run it whenever you add or change documents in data/.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config -- the knobs you will be asked about in an interview. See README.md.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800        # characters per chunk (roughly 150-200 tokens)
CHUNK_OVERLAP = 150     # characters shared between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, local, free, fast

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size character windows with overlap.

    Deliberately simple (no sentence awareness) so the tradeoff is easy to
    explain: real systems often chunk on sentence/paragraph boundaries or
    use token-based splitters. See the README's tradeoffs section.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


def main() -> None:
    INDEX_DIR.mkdir(exist_ok=True)

    docs = sorted(DATA_DIR.glob("*.txt"))
    if not docs:
        raise SystemExit(f"No .txt documents found in {DATA_DIR}")

    chunks: list[dict] = []
    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({"source": doc_path.name, "chunk_id": i, "text": chunk})

    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads once, then cached)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)  # so IndexFlatIP == cosine similarity

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "chunks.faiss"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"model": EMBEDDING_MODEL, "chunks": chunks}, f, indent=2)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={embeddings.shape[1]}) "
          f"and metadata to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
