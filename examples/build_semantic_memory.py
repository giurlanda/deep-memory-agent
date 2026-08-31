"""Build a memory tree with the manager agent, index it, then recall by meaning.

This is the semantic counterpart of `basic_usage.py`: the manager records a few
things in its own words, the tree is indexed into a hybrid Qdrant collection,
and a read-only agent answers questions phrased *differently* from how the
entries were written — which is exactly what `memory_search` cannot do.

Run:
    docker run -p 6333:6333 qdrant/qdrant
    uv sync --extra semantic --group examples
    export ANTHROPIC_API_KEY=...
    uv run python examples/build_semantic_memory.py

Embeddings come from a local OpenAI-compatible server (LM Studio here), so
indexing costs nothing and never leaves the machine; only the agents talk to a
remote model. That split is the usual one in practice.
"""

from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from deep_memory_agent import (
    create_memory_manager_agent,
    create_memory_search_agent,
    ingest_semantic_index,
)

MODEL = "claude-sonnet-5"
MEMORY_DIR = Path(__file__).parent / "semantic-memory"

COLLECTION = "semantic-memory"
QDRANT_URL = "http://localhost:6333"

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# What the manager is told, in the wording a user would actually use.
FACTS = [
    (
        "Remember that we always give the Enterprise discount to customers who "
        "have been under contract for more than three years."
    ),
    (
        "Write down the procedure for rolling back a deploy when a database "
        "migration is only half applied: put the release on hold, run the down "
        "migration for the batch that landed, restore the previous image, and "
        "only then re-open traffic."
    ),
    (
        "Note this mistake so it does not happen again: last week we quoted "
        "Acme using the previous year's price list."
    ),
]

# What is asked afterwards, deliberately sharing almost no words with the above.
QUESTIONS = [
    "What price reductions do we offer to long-standing clients?",
    "What should I do if an update to the database gets stuck during a release?",
    "Have we ever made errors when preparing a quote?",
]


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


def say(agent: CompiledStateGraph, message: str, thread: str) -> str:
    """Send one message to an agent and return its final reply."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread}},
    )
    return result["messages"][-1].content


def main() -> None:
    embeddings = build_embeddings()
    store = build_store(embeddings)

    # The manager gets `semantic_ingest` as well as the write tools, because it
    # is the single writer of the tree and so the only agent that can keep a
    # derived index in step with it.
    manager = create_memory_manager_agent(
        MODEL,
        memory_dir=MEMORY_DIR,
        embeddings=embeddings,
        vector_store=store,
        search_k=8,
    )

    print("=== recording\n")
    for number, fact in enumerate(FACTS, start=1):
        print(say(manager, fact, thread=f"write-{number}"))

    # Ingestion is explicit, so this is also the check on whether the manager
    # kept the index current on its own: "No update needed" means it did.
    print("\n=== ingest (safety net after the agent's own)\n")
    print(ingest_semantic_index(embeddings, store, memory_dir=MEMORY_DIR).summary())

    print("\n=== ingest again (nothing should change)\n")
    print(ingest_semantic_index(embeddings, store, memory_dir=MEMORY_DIR).summary())

    # The recall agent gets `semantic_search` but never `semantic_ingest`: the
    # filesystem deny rule cannot guard a call to an external vector store, so
    # withholding the tool is the only real read-only guarantee.
    recall = create_memory_search_agent(
        MODEL,
        memory_dir=MEMORY_DIR,
        embeddings=embeddings,
        vector_store=store,
        search_k=8,
    )

    for number, question in enumerate(QUESTIONS, start=1):
        print(f"\n=== {question}\n")
        print(say(recall, question, thread=f"recall-{number}"))

    print(f"\nMemory written to {MEMORY_DIR.resolve()}:")
    for path in sorted(MEMORY_DIR.rglob("*.md")):
        print(f"  {path.relative_to(MEMORY_DIR)}")


if __name__ == "__main__":
    main()
