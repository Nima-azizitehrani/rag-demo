# RAG Demo — Food Safety Q&A

A small, runnable retrieval-augmented generation (RAG) demo built for
interview preparation. It answers questions over four public Wikipedia
documents about food safety, HACCP, foodborne illness, and cold-chain
logistics — no company data, no private data (sources in `data/SOURCES.md`).

## What RAG is, in five lines

1. An LLM only knows what it was trained on; RAG lets it answer from *your*
   documents instead.
2. Offline, you split documents into chunks, turn each chunk into an embedding
   (a vector of numbers capturing its meaning), and store the vectors in an index.
3. At question time, you embed the question the same way and retrieve the
   chunks whose vectors are closest — i.e., most semantically similar.
4. You paste those chunks into the prompt as context and instruct the model to
   answer only from them.
5. Done well, you get current, citable, domain-specific answers without
   retraining any model.

## Architecture of this demo

```
data/*.txt ──> ingest.py ──> chunk (800 chars, 150 overlap)
                          ──> embed (all-MiniLM-L6-v2, local, 384-dim)
                          ──> FAISS IndexFlatIP on normalized vectors
                              (= cosine similarity search)
                          ──> saved to index/chunks.faiss + index/chunks.json

question ────> ask.py ─────> embed question (same model)
                          ──> FAISS top-4 chunks
                          ──> if ANTHROPIC_API_KEY or OPENAI_API_KEY is set:
                                 grounded prompt -> LLM answer + source list
                              else:
                                 RETRIEVAL MODE — prints the top passages
                                 themselves (great for inspecting quality)
```

Key design choice to be able to explain: the *same* embedding model must be
used at ingest time and query time, or the vectors live in different spaces
and similarity is meaningless. The model name is stored in the index metadata
for exactly that reason.

## Run it end to end

```bash
cd rag-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python ingest.py        # builds index/ from data/ (first run downloads the ~90 MB model)

python ask.py "What are the seven principles of HACCP?"
python ask.py "What is the temperature danger zone for bacteria?"
python ask.py "How do I renew my driver's license?"     # not in the docs — watch what happens

# Optional: real LLM answers instead of retrieval mode
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY=...
pip install anthropic               # or: pip install openai
python ask.py "What are the seven principles of HACCP?"
```

Model names are configurable: `RAG_ANTHROPIC_MODEL` (default
`claude-haiku-4-5`) and `RAG_OPENAI_MODEL` (default `gpt-4o-mini`).
Chunk size and overlap are constants at the top of `ingest.py`; top-k is a
constant in `ask.py`. Change them, re-run ingest, and compare retrieval —
that experiment is the fastest way to build intuition.

> **Verified end-to-end on 2026-10-25** (Python 3.12, sentence-transformers
> 3.0.1, faiss-cpu 1.8.0): ingest produced 33 chunks from the 4 documents.
> "What are the seven principles of HACCP?" retrieved the right chunks from
> `haccp.txt` at similarity 0.47–0.59; an out-of-corpus question ("How do I
> renew my driver's license in Georgia?") topped out at 0.073 — a clear,
> visible gap, which is what a production similarity threshold would use to
> trigger the "documents don't cover this" response. Troubleshooting: on
> machines where `/tmp` is small, set `TMPDIR` to a folder on disk before
> `pip install` (the PyTorch wheel is ~550 MB); if the model download from
> Hugging Face fails through a proxy, retry with `HF_HUB_DISABLE_XET=1`.

## Tradeoffs you should be able to talk about

### Chunk size and overlap
Too small (say 100 characters) and each chunk loses the surrounding context —
you retrieve a sentence fragment that mentions a number but not what the
number belongs to. Too large (say 4,000 characters) and one chunk covers
several topics: its embedding becomes an average of everything, relevance
dilutes, and you burn prompt tokens on text the model mostly ignores.
Overlap (here 150 of 800 characters) exists so a fact sitting on a chunk
boundary survives intact in at least one chunk; the cost is duplicated
content and a slightly bigger index. There is no universally right answer —
you tune it against real questions your users ask.

### Embedding model: local MiniLM vs API embeddings
This demo uses `all-MiniLM-L6-v2` locally: free, private (documents never
leave the machine — a real argument when the corpus is internal company
data), fast, and good enough for a small corpus. API embeddings (e.g.
OpenAI's) are generally higher quality, handle longer inputs, and need zero
local compute — but they cost per token, add a network dependency, and send
your text to a third party. The deciding factors are corpus sensitivity,
scale, and measured retrieval quality, not fashion.

### Vector store: FAISS vs Chroma vs pgvector
- **FAISS** (this demo): a library, not a database. Extremely fast
  in-process search, zero infrastructure. No built-in metadata filtering,
  persistence, or multi-user access — you manage files yourself. Right for
  prototypes, batch jobs, and embedded use.
- **Chroma**: a real (small) database built for this use case — persistent,
  stores documents + metadata alongside vectors, filters by metadata, simple
  API. Right for most small-to-medium application backends.
- **pgvector**: an extension that puts vectors inside Postgres. Choose it
  when your data already lives in Postgres and you want one system to back
  up, secure, and join vector search against relational data. Search speed
  is fine at moderate scale; you gain operational simplicity, not raw speed.
The interview-grade answer: the vector store is rarely the bottleneck —
retrieval quality (chunking, embeddings, query handling) is.

### Grounding and hallucination control
RAG reduces hallucination; it does not eliminate it. The controls, in order
of importance: (1) a system prompt that orders the model to answer *only*
from the provided context and to say the documents don't cover something
when they don't (see `GROUNDING_PROMPT` in `ask.py`); (2) returning source
citations with every answer so a human can verify in one click; (3) decent
retrieval — a grounded prompt over wrong chunks produces confidently wrong
answers with citations, which is worse than an honest "I don't know";
(4) a similarity threshold in production, so weak matches route to
"I don't know" instead of being forced into the prompt.

### Retrieval failure modes
- The answer exists but is phrased differently than the question (vocabulary
  mismatch) — semantic embeddings help, hybrid keyword+vector search helps more.
- The answer spans multiple chunks or documents and top-k returns only part
  of it — raise k, add overlap, or summarize per document first.
- The chunker split a table, list, or procedure mid-way, so the key half is
  in the next chunk — structure-aware chunking beats fixed windows here.
- Two documents disagree (an old SOP vs the new one) and both get retrieved —
  that's a data governance problem; metadata (date, version, source) and
  filtering exist precisely for this.
- The question can't be answered from the corpus at all — the correct system
  behavior is to say so, and your demo should show that, not hide it.

### When RAG is the WRONG tool
- **Structured data** (rebates, sales, inventory): the answer lives in rows
  and columns. Use SQL / a BI tool, or text-to-SQL with guardrails. Embedding
  a spreadsheet and asking a vector store "what was Q3 volume?" is how you
  get plausible, wrong numbers.
- **Stable behavior, tone, or format**: if you want the model to always
  respond in a house style or follow a fixed procedure, that's prompt design
  (or fine-tuning at real scale), not retrieval — the knowledge isn't changing.
- **Tiny, stable corpora**: if the whole document set fits comfortably in
  the context window (a 20-page policy manual), skip the vector store and
  put the text straight in the prompt. Simpler, and retrieval can't fail.
- **Exact-lookup tasks**: part numbers, regulation clauses, keyword search —
  classic search (BM25/full-text) is more precise than semantic similarity.

## Five likely interview questions, with model answers

**1. "Explain RAG to me like I'm a business user."**
"You have documents the AI has never seen — our manuals, our policies. Instead
of retraining the AI, we convert those documents into a searchable form.
When someone asks a question, we find the few passages that actually answer
it and hand them to the AI along with the question, with instructions to
answer from those passages and cite them. The AI provides the language
skills; our documents provide the facts."

**2. "Your demo uses FAISS. What would change in production?"**
"Three things. Persistence and metadata: I'd move to Chroma or pgvector so
chunks carry source, date, and version, and I can filter instead of just
ranking. Access control: retrieved chunks must respect the user's
permissions, or the system leaks documents through answers. And evaluation:
I'd build a small set of real questions with known answers and measure
retrieval hit-rate and answer faithfulness every time I change chunking,
the embedding model, or the documents. The index library is the easy part;
the lifecycle around it is the job."

**3. "How do you stop the model from making things up?"**
"Layered. The prompt restricts the model to the retrieved context and
requires an explicit 'not covered' response. Answers carry citations so
users can verify cheaply. Below a similarity threshold the system refuses
rather than answering from weak matches. And I monitor real failures —
when the model hallucinates, the root cause is usually retrieval returning
the wrong chunk, not the prompt, so that's where fixes go first."

**4. "Why not just fine-tune a model on our documents?"**
"Fine-tuning teaches behavior and style; it's a poor way to store facts.
It goes stale the day a document changes, it's expensive to redo, and the
model still can't cite where an answer came from. If our problem is 'answer
questions from documents that change,' retrieval is the right tool. I'd
fine-tune when the problem is 'always respond in this format / follow this
procedure' — a stable behavioral target."

**5. "Where would RAG help at a company like ours, and where wouldn't it?"**
"Good fits: a Member or employee asking natural-language questions across
hundreds of SOPs, program guides, and policy PDFs — documents that change
and need citations. Bad fit: anything numeric and transactional. 'What were
last quarter's rebate earnings by member?' belongs in SQL against the data
warehouse, where the number is exact and auditable — I'd never let a vector
store near that. The judgment is matching the tool to whether the source of
truth is prose or structured data."
