"""
ask.py -- ask a question against the index built by ingest.py.

Usage:  python ask.py "What are the seven principles of HACCP?"

Behavior:
  * Embeds the question with the same local model used at ingest time,
    retrieves the top-k most similar chunks from FAISS.
  * If ANTHROPIC_API_KEY or OPENAI_API_KEY is set in the environment, the
    retrieved chunks are sent to that LLM as context and you get a grounded
    answer. The SDKs are optional imports -- the demo runs fine without them.
  * Without an API key it runs in RETRIEVAL MODE: it prints the top passages
    (with source document names) that an LLM would answer from. That is not a
    failure state -- inspecting raw retrieval is how you debug RAG quality.
"""

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K = 4                                   # chunks retrieved per question
ANTHROPIC_MODEL = os.environ.get("RAG_ANTHROPIC_MODEL", "claude-haiku-4-5")
OPENAI_MODEL = os.environ.get("RAG_OPENAI_MODEL", "gpt-4o-mini")

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"

GROUNDING_PROMPT = """You are answering questions about food safety using ONLY the
context passages below. Rules:
- Answer only from the provided context. Do not use outside knowledge.
- If the answer is not in the context, say plainly that the documents do not
  cover it. Do not guess or fill gaps.
- Cite the source document name(s) you used, like [haccp.txt].

Context:
{context}

Question: {question}"""


def load_index():
    index_path = INDEX_DIR / "chunks.faiss"
    meta_path = INDEX_DIR / "chunks.json"
    if not index_path.exists() or not meta_path.exists():
        raise SystemExit("No index found. Run `python ingest.py` first.")
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta["chunks"], meta["model"]


def retrieve(index, chunks, model_name: str, question: str, k: int = TOP_K):
    model = SentenceTransformer(model_name)
    q = model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        c = chunks[idx]
        results.append({"score": float(score), "source": c["source"],
                        "chunk_id": c["chunk_id"], "text": c["text"]})
    return results


def answer_with_llm(question: str, results: list[dict]) -> str | None:
    """Call an LLM if a key is configured; return None if no provider is set."""
    context = "\n\n".join(
        f"[{r['source']} #{r['chunk_id']}]\n{r['text']}" for r in results)
    prompt = GROUNDING_PROMPT.format(context=context, question=question)

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # optional dependency
        except ImportError:
            print("(ANTHROPIC_API_KEY is set but the `anthropic` package is not "
                  "installed; falling back to retrieval mode.)")
            return None
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=600,
            messages=[{"role": "user", "content": prompt}])
        return "".join(block.text for block in msg.content if block.type == "text")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # optional dependency
        except ImportError:
            print("(OPENAI_API_KEY is set but the `openai` package is not "
                  "installed; falling back to retrieval mode.)")
            return None
        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

    return None


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python ask.py "your question"')
    question = " ".join(sys.argv[1:])

    index, chunks, model_name = load_index()
    results = retrieve(index, chunks, model_name, question)

    answer = answer_with_llm(question, results)
    if answer is not None:
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n=== SOURCES USED ===")
        for r in results:
            print(f"  - {r['source']} (chunk {r['chunk_id']}, score {r['score']:.3f})")
        return

    print("\n=== RETRIEVAL MODE (no LLM API key set) ===")
    print("Top passages an LLM would answer from:\n")
    for rank, r in enumerate(results, 1):
        print(f"--- #{rank}  {r['source']}  chunk {r['chunk_id']}  "
              f"similarity {r['score']:.3f} ---")
        print(r["text"])
        print()
    print("Note: with ANTHROPIC_API_KEY or OPENAI_API_KEY set, ask.py sends "
          "these passages to the model and prints a grounded answer instead.")


if __name__ == "__main__":
    main()
