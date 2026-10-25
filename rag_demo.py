"""
A tiny RAG demo in one file.

RAG stands for Retrieval-Augmented Generation. The idea:
  1. Take a few documents.
  2. When someone asks a question, find the pieces of text that best
     match the question.
  3. (In a full system) hand those pieces to an AI model and let it
     answer using only them.

This demo does steps 1 and 2 - the retrieval half - and prints the
pieces it found, with a score for each. No API key is needed: the
embedding model runs on this computer.

The documents are four public Wikipedia articles about food safety
(see data/SOURCES.md). No company or private data.

Run:
    pip install -r requirements.txt
    python rag_demo.py
"""
import glob
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- Settings ----------------
MODEL_NAME = "all-MiniLM-L6-v2"  # a small, free embedding model
CHUNK_SIZE = 800                 # characters per piece of text
OVERLAP = 150                    # characters two neighbour pieces share
TOP_K = 4                        # pieces to show for each question
MIN_SCORE = 0.20                 # below this, the documents don't cover it

# The embedding model. It turns text into a list of 384 numbers that
# capture the meaning of the text. Similar meanings -> similar numbers.
model = SentenceTransformer(MODEL_NAME)


def embed(texts):
    """Turn a list of texts into number-lists (embeddings)."""
    vectors = model.encode(texts)
    # Normalise, so that comparing two vectors gives a similarity
    # score between -1 and 1 (this is called cosine similarity).
    faiss.normalize_L2(vectors)
    return vectors.astype("float32")


# ---------------- Step 1: read the documents ----------------
def read_documents(folder="data"):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


# ---------------- Step 2: cut each document into pieces ----------------
def cut_into_pieces(text):
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP  # step forward, keeping the overlap
    return pieces


# ---------------- Step 3: build the searchable index ----------------
def build_index():
    chunks = []   # the pieces of text
    sources = []  # which document each piece came from
    for name, text in read_documents():
        for piece in cut_into_pieces(text):
            chunks.append(piece)
            sources.append(name)
    print(f"Read {len(sources)} pieces from the documents.")

    index = faiss.IndexFlatIP(384)      # FAISS stores the numbers...
    index.add(embed(chunks))            # ...and finds the closest ones fast
    return index, chunks, sources


# ---------------- Step 4: answer questions ----------------
def ask(index, chunks, sources, question):
    question_vector = embed([question])
    scores, ids = index.search(question_vector, TOP_K)

    if scores[0][0] < MIN_SCORE:
        print("  The documents don't seem to cover this question.")
        print(f"  (Best score was only {scores[0][0]:.2f}.)")
        return

    for score, i in zip(scores[0], ids[0]):
        print(f"\n  [{score:.2f}] from {sources[i]}:")
        print("  " + chunks[i].replace("\n", " ")[:300] + "...")


if __name__ == "__main__":
    index, chunks, sources = build_index()
    print("\nAsk a question about food safety (or type 'quit').")
    while True:
        try:
            q = input("\n> ")
        except EOFError:
            break
        if q.strip().lower() == "quit":
            break
        if q.strip():
            ask(index, chunks, sources, q)
