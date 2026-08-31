# Semantic search

`memory_search` is a case-insensitive substring test over an entry's summary,
body and tags. That is enough when the question shares words with the entry that
answers it — and useless when it does not:

- A rule stored as *"always apply the Enterprise discount to customers with more
  than 3 years of contract"* is invisible to *"what discounts exist for
  long-standing customers?"*
- A procedure summarised as *"rollback of a deploy with a half-applied
  migration"* does not surface for *"what do I do if a database update hangs
  during a release?"*
- An error described as *"quoted with last year's price list"* does not answer
  *"quoting mistakes"* if that word never appears in the text.

The caller never knows the wording an entry was written with, so this is not an
edge case — it is the normal way a memory lookup fails. The semantic index adds
an embedding-based route to the same entries, covering all seven categories.

## The index is derived data

The markdown files stay the single source of truth. The index holds nothing they
do not, can be deleted and rebuilt at any time, and losing it costs the ability
to search by meaning until the next ingest — never a fact. That is the same
principle the router `index.md` files follow, and it is why no vector store is
pinned by this package: which one to use is yours to decide, and memory stays a
directory of markdown either way.

## Installation

Semantic search needs one extra dependency, a text splitter:

```bash
pip install "deep-memory-agent[semantic]"
# or: uv add "deep-memory-agent[semantic]"
```

The embedding model and the vector store are not pinned — bring your own. The
only requirement on the store is that `add_documents(docs, ids=[...])` **upserts
on a repeated id** rather than duplicating it, which is what makes re-ingesting
an unchanged entry a no-op. Chroma, Qdrant and FAISS-with-explicit-ids all do;
the abstract `VectorStore` interface does not guarantee it.

## Turning it on

Hand either factory an `embeddings` model and a `vector_store` and the tools
appear:

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from deep_memory_agent import create_memory_manager_agent

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

manager = create_memory_manager_agent(
    "claude-sonnet-5",
    memory_dir="./memory",
    embeddings=embeddings,
    vector_store=InMemoryVectorStore(embeddings),
)
```

Passing only one of the two raises `ValueError`: half a configuration is a
mistake, and silently running without an index would leave you believing memory
is searchable by meaning when it is not.

Which tools each agent gets is deliberate:

| Agent | `semantic_search` | `semantic_ingest` |
| --- | --- | --- |
| [`create_memory_manager_agent`][deep_memory_agent.create_memory_manager_agent] | yes | yes |
| [`create_memory_search_agent`][deep_memory_agent.create_memory_search_agent] | yes | **no** |

The recall agent never gets the ingest tool.
[`READ_ONLY_MEMORY_PERMISSIONS`][deep_memory_agent.READ_ONLY_MEMORY_PERMISSIONS]
denies writes at the filesystem middleware, but it has no say over a tool that
talks to an external vector store — so withholding the tool is the only real
guarantee that the recall agent stays read-only.

## Keeping the index current

Ingestion is **explicit**. Unlike the router indexes, which `memory_write`
updates on its own, nothing refreshes the semantic index for you: the manager
calls `semantic_ingest` after writing, or a job outside the agent calls
[`ingest_semantic_index`][deep_memory_agent.ingest_semantic_index].

The cost of that choice is a window in which the index trails the files, widest
right after a supersession. Both prompts say so — the manager's tells it to
ingest after every write, the reader's warns that the freshest state of memory is
the files, not the index.

For a deterministic refresh — a cron entry, a pre-commit hook, the rebuild step
of a deployment — skip the model entirely:

```python
from deep_memory_agent import ingest_semantic_index

report = ingest_semantic_index(embeddings, vector_store, memory_dir="./memory")
print(report.summary())
# "No update needed: 42 entries already match the index."
```

Re-running an ingest with nothing changed leaves the vector store byte-for-byte
as it was and says so explicitly, so it is safe to call on a schedule.

## What gets indexed

The unit is an **entry**, not a file — a monthly episodic shard holds many, and
indexing the file would blur them together. Each entry becomes one chunk, unless
its body passes `chunk_size` (800 characters by default), which in practice only
happens to procedures with their fixed sections. Chunk ids are derived with
`uuid5` from `entry_id::chunk_index`, so the same entry always lands under the
same id — that is what makes a re-ingest an update instead of a duplicate — and
they are UUID-shaped, which stores like Qdrant require.

Every chunk carries the entry's whole frontmatter as metadata, which is what
`semantic_search` filters on:

| Filter | Metadata field |
| --- | --- |
| `kind`, `category` | `kind`, `category` |
| `tags` (all must be present) | `tags` |
| `source` | `source` |
| `min_confidence` | `confidence` |
| `created_after` / `created_before` | `created` |
| `include_superseded` | `is_active` |

`include_superseded` defaults to `False`, matching `memory_search`: an agent
searching by meaning does not expect a retired fact to surface unless it asked
for history.

Filters run client-side by default, over a candidate set widened by `over_fetch`.
If your store can filter on metadata itself, give `SemanticConfig` a
`filter_builder` and they run server-side instead:

```python
from deep_memory_agent import SemanticConfig, create_memory_manager_agent

config = SemanticConfig(
    filter_builder=lambda active: {"must": [...]},  # your store's filter object
)
```

Hits are **chunks**, so an entry with a long body can occupy more than one of the
`k` slots; regroup them by the `entry_id` each carries. To read the entry behind
a hit in full, use `memory_get`, which takes an entry id and returns just that
entry — `memory_read` would hand you every other entry in the same file.

## The manifest

Ingest state lives at `/memory/.semantic-manifest.json`: one row per entry with
its digest, its file, its category and the ids its chunks went in under. It sits
inside the tree, so it travels with it and is versioned by the same git
repository, but it is not a `.md` file, so `MemoryStore` never globs it into a
lexical search. A missing or malformed manifest is not an error — it just makes
the next ingest a full one.

The manifest is what makes an ingest incremental, and what lets it notice
entries that disappeared. Nothing in this package deletes an entry — `MemoryStore`
has no `delete`, and retiring an entry rewrites it rather than removing it — so
that path exists for changes made outside the agent's API: a shard file deleted
by hand, a block of entries edited out, a `git checkout` of `/memory/` to an
earlier state. Without it, a derived index could outlive its source and keep
answering with entries the files no longer hold.

A scoped ingest never prunes what it did not look at: running
`semantic_ingest(category="procedure")` leaves the manifest rows of `facts`
exactly where they are.

## Cost note

A change to metadata alone re-embeds the whole entry: when an entry is retired,
its digest changes and its chunks are deleted and rewritten, even though the
indexed text did not move. `add_documents` generally offers no way to update a
vector's metadata in place. At the write volume this system sees that is fine;
it is worth revisiting if it stops being.
