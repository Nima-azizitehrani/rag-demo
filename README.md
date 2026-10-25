# RAG Demo

A small demo of **RAG (Retrieval-Augmented Generation)** in one Python file: `rag_demo.py`.

## What it does

1. Reads four public Wikipedia articles about food safety (in `data/`).
2. Cuts them into small pieces that overlap a little, so ideas are not cut in half.
3. Turns each piece into a list of numbers — an *embedding* — using a small model
   (`all-MiniLM-L6-v2`) that runs on your own computer. Pieces with similar
   meaning get similar numbers. No API key, nothing leaves the machine.
4. Stores the numbers in a **FAISS** index, which can find the closest ones fast.
5. When you ask a question, it turns the question into numbers the same way and
   prints the pieces that match best, each with a similarity score.

That is the *retrieval* half of RAG. In a full system, those pieces would then be
handed to an AI model, which writes the final answer using only them.

## Run it

```bash
pip install -r requirements.txt
python rag_demo.py
```

Then type a question, for example: *What are the seven principles of HACCP?*

## Things worth noticing

- **The score gap is the point.** The HACCP question matches at about 0.5–0.6.
  An unrelated question ("How do I renew my driver's license in Georgia?")
  matches at only ~0.10, and the demo says the documents don't cover it. That
  gap is how a real system knows when it should refuse instead of guessing.
- **Chunk size 800, overlap 150.** Small enough that each piece is about one
  idea; overlapped so an idea sitting on a boundary is not lost.
- **Local model + FAISS, not an API + hosted database.** Local is free and
  private, and fine at this size. A hosted service scales to millions of
  documents, but costs money and sends your data out. Which one is right
  depends on the job.

## Files

- `rag_demo.py` — everything, in one file
- `data/` — the four articles; sources listed in `data/SOURCES.md`
- `requirements.txt` — the libraries needed
