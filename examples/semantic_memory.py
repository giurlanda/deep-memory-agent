"""Index an existing memory tree for semantic + keyword search, then query it.

Run `examples/build_semantic_memory.py` (or `examples/basic_usage.py`) first so
there is a tree to index, start a local Qdrant, then:

    docker run -p 6333:6333 qdrant/qdrant
    uv sync --extra semantic --group examples
    export ANTHROPIC_API_KEY=...
    uv run python examples/semantic_memory.py

Three things this shows that the README cannot:

- ingestion runs as a plain function, before any agent exists, and reports what
  it did — a second run reports that everything was already up to date, which is
  what makes it safe to put on a cron;
- the same question run through both searches side by side, with no model in the
  loop, so the gap the index closes is visible rather than asserted;
- the store is in hybrid mode, so the dense half and BM25 both see the query.
  Qdrant is one choice among many: any LangChain `VectorStore` works, and the
  library pins none — the only requirement is that it upserts on a repeated id.
"""

from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from deep_memory_agent import (
    MemoryStore,
    SemanticIndex,
    build_memory_backend,
    create_memory_search_agent,
    ingest_semantic_index,
)

MODEL = "claude-sonnet-5"
MEMORY_DIR = Path(__file__).parent / "semantic-memory"

COLLECTION = "semantic-memory"
QDRANT_URL = "http://localhost:6333"

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

QUESTION = "What price reductions do we offer to long-standing clients?"

# The keyword someone asking that question would plausibly reach for. Neither it
# nor the question shares a word with the entry that answers them, which is the
# whole point: the asker does not know the vocabulary the entry was written in.
KEYWORD = "reductions"


def build_embeddings() -> OpenAIEmbeddings:
    """Return the local embedding client used for both indexing and querying."""
    # LM Studio's /v1/embeddings only accepts text; without
    # `check_embedding_ctx_length=False` the LangChain client pre-tokenises and
    # sends token-id arrays, which the server rejects with a 400.
    return OpenAIEmbeddings(
        model="text-embedding-embeddinggemma-300m",
        base_url="http://127.0.0.1:1234/v1",
        api_key="no-key",
        check_embedding_ctx_length=False,
    )


def build_store(embeddings: OpenAIEmbeddings) -> QdrantVectorStore:
    """Create the collection if needed and return a hybrid store over it."""
    client = QdrantClient(url=QDRANT_URL)
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR: qmodels.VectorParams(
                    size=len(embeddings.embed_query("probe")),
                    distance=qmodels.Distance.COSINE,
                )
            },
            # Modifier.IDF is mandatory for BM25: Qdrant computes the IDF over
            # the corpus itself, and without it the sparse half scores nothing.
            sparse_vectors_config={
                SPARSE_VECTOR: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
            },
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        vector_name=DENSE_VECTOR,
        sparse_embedding=sparse,
        sparse_vector_name=SPARSE_VECTOR,
        retrieval_mode=RetrievalMode.HYBRID,
    )


def lexical(store: MemoryStore, query: str) -> None:
    """Print what the substring search finds for one query."""
    print(f"\n=== lexical: memory_search({query!r})\n")
    hits = store.search(query, limit=3)
    if not hits:
        print("  (nothing — no entry contains that text)")
    for hit in hits:
        print(
            f"  {hit.entry.entry_id} ({hit.entry.category.value}) — {hit.entry.summary}"
        )


def compare(index: SemanticIndex, store: MemoryStore, question: str) -> None:
    """Run one question through both searches and print what each finds.

    The lexical search is given two fair shots — the whole question, and the
    single keyword its own docstring tells a model to prefer — so the comparison
    is not against a strawman. Both miss for the same reason: a substring test
    needs the asker to have guessed the entry's wording.
    """
    lexical(store, question)
    lexical(store, KEYWORD)

    print(f"\n=== semantic: semantic_search({question!r})\n")
    for hit in index.search(question, k=3):
        score = f"{hit['score']:.4f}" if hit["score"] is not None else "n/a"
        print(f"  [{score}] {hit['entry_id']} ({hit['category']}) — {hit['summary']}")


def main() -> None:
    if not MEMORY_DIR.is_dir():
        msg = (
            f"{MEMORY_DIR} does not exist — run examples/build_semantic_memory.py first"
        )
        raise SystemExit(msg)

    embeddings = build_embeddings()
    store = build_store(embeddings)

    # Deterministic ingestion: no model decides whether this runs.
    print("=== ingest\n")
    print(ingest_semantic_index(embeddings, store, memory_dir=MEMORY_DIR).summary())

    print("\n=== ingest again (nothing should change)\n")
    print(ingest_semantic_index(embeddings, store, memory_dir=MEMORY_DIR).summary())

    # `for_deep_agent=False` is what makes the backend usable outside an agent:
    # the default wiring parks non-memory paths in LangGraph thread state, which
    # only exists inside a graph execution.
    memory = MemoryStore(build_memory_backend(MEMORY_DIR, for_deep_agent=False))
    compare(SemanticIndex(embeddings, store, memory), memory, QUESTION)

    recall = create_memory_search_agent(
        MODEL,
        memory_dir=MEMORY_DIR,
        embeddings=embeddings,
        vector_store=store,
        search_k=8,
    )

    result = recall.invoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
        config={"configurable": {"thread_id": "semantic-1"}},
    )
    print(f"\n=== agent answer: {QUESTION}\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
