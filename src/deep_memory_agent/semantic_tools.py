"""Semantic ingestion and search, exposed as agent tools.

Two tools come out of one factory, because they are the two halves of one index:
`semantic_ingest` writes the memory tree into a vector store, `semantic_search`
queries it by meaning. The manager gets both — it is the single writer of
`/memory/`, so it is the one that has to keep the index in step. The recall
agent gets only the search: withholding the ingest tool is the *only* guarantee
of read-only here, since
[`READ_ONLY_MEMORY_PERMISSIONS`][deep_memory_agent.agents.READ_ONLY_MEMORY_PERMISSIONS]
guards the filesystem and has no say over a call to an external vector store.

The embedding model, the vector store and the memory tree are captured in the
closure, never taken as tool arguments: the model chooses what to index and what
to look for, not where to read from or where to write to.

Both operations are also plain Python —
[`ingest_semantic_index`][deep_memory_agent.semantic_tools.ingest_semantic_index]
is the same code path with no model in the middle, for a cron entry, a
pre-commit hook or a maintenance script.

Requires the optional `semantic` extra::

    pip install "deep-memory-agent[semantic]"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from deep_memory_agent.backends import resolve_backend
from deep_memory_agent.layout import MemoryCategory, MemoryKind
from deep_memory_agent.semantic_index import IngestReport, SemanticConfig, SemanticIndex
from deep_memory_agent.store import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.embeddings import Embeddings
    from langchain_core.tools import BaseTool
    from langchain_core.vectorstores import VectorStore

__all__ = [
    "SEMANTIC_INGEST_TOOL_NAME",
    "SEMANTIC_SEARCH_TOOL_NAME",
    "SemanticTools",
    "create_semantic_tools",
    "ingest_semantic_index",
]

SEMANTIC_INGEST_TOOL_NAME = "semantic_ingest"
SEMANTIC_SEARCH_TOOL_NAME = "semantic_search"

_MAX_K = 25
_K_DESCRIPTION = (
    "How many excerpts to return. A long entry can contribute more than one, "
    "so this is not a count of distinct entries."
)
_DEFAULT_SNIPPET_CHARS = 500

_KIND_VALUES = ", ".join(f"`{kind.value}`" for kind in MemoryKind)
_CATEGORY_VALUES = ", ".join(f"`{category.value}`" for category in MemoryCategory)


class SemanticIngestArgs(BaseModel):
    """Arguments of the ingestion tool."""

    kind: str | None = Field(
        default=None,
        description=(
            f"Restrict the ingest to one memory kind ({_KIND_VALUES}). Leave it "
            "out to index the whole tree, which is the usual case — unchanged "
            "entries are skipped, so a full run is cheap. A scoped run never "
            "removes entries outside its scope from the index."
        ),
    )
    category: str | None = Field(
        default=None,
        description=(
            f"Restrict the ingest to one category ({_CATEGORY_VALUES}). Same "
            "caveat as `kind`."
        ),
    )
    only_modified: bool = Field(
        default=True,
        description=(
            "Index only what changed since the last run. Keep it true for the "
            "routine update after writing; set it to false to rebuild the index "
            "from scratch."
        ),
    )


class SemanticSearchArgs(BaseModel):
    """Arguments of the search tool."""

    query: str = Field(
        ...,
        description=(
            "What you are looking for, in natural language. Unlike "
            "`memory_search` this matches meaning, so ask the question the way "
            "you would ask a person rather than guessing the entry's wording."
        ),
    )
    k: int = Field(default=5, ge=1, le=_MAX_K, description=_K_DESCRIPTION)
    kind: str | None = Field(
        default=None,
        description=f"Restrict to one memory kind ({_KIND_VALUES}).",
    )
    category: str | None = Field(
        default=None,
        description=f"Restrict to one category ({_CATEGORY_VALUES}).",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Only return entries carrying all of these tags.",
    )
    source: str | None = Field(
        default=None,
        description=(
            "Only return entries recorded with this source, e.g. "
            "`user_message`, `user_feedback`, `tool_result`, `consolidation`."
        ),
    )
    min_confidence: str | None = Field(
        default=None,
        description=(
            "Only return entries at or above this confidence: `low`, `medium`, `high`."
        ),
    )
    created_after: str | None = Field(
        default=None,
        description="ISO date or timestamp; only entries created at or after it.",
    )
    created_before: str | None = Field(
        default=None,
        description="ISO date or timestamp; only entries created at or before it.",
    )
    include_superseded: bool = Field(
        default=False,
        description=(
            "Also return entries a newer one replaced. Use only when the "
            "history itself is what matters."
        ),
    )


def _search_args_schema(default_k: int) -> type[BaseModel]:
    """Return the search schema with `k` defaulted to this index's setting.

    The schema is what the model sees, so a `search_k` that only reached the
    Python function would be silently overridden the moment the model omitted
    `k` — which is most of the time.

    Args:
        default_k: Number of entries to return when the model does not say.

    Returns:
        A subclass of `SemanticSearchArgs` carrying that default.
    """
    return create_model(
        "SemanticSearchArgs",
        __base__=SemanticSearchArgs,
        k=(int, Field(default=default_k, ge=1, le=_MAX_K, description=_K_DESCRIPTION)),
    )


@dataclass
class SemanticTools:
    """The tools of one index, plus the functions underneath them.

    Attributes:
        ingest_tool: The `semantic_ingest` tool.
        search_tool: The `semantic_search` tool.
        index: The
            [`SemanticIndex`][deep_memory_agent.semantic_index.SemanticIndex]
            both tools drive, for callers who want it directly.
    """

    ingest_tool: BaseTool
    search_tool: BaseTool
    index: SemanticIndex

    @property
    def ingest(self) -> Callable[..., IngestReport]:
        """The ingestion function, without the tool wrapper."""
        return self.index.ingest

    @property
    def search(self) -> Callable[..., list[dict[str, Any]]]:
        """The search function, without the tool wrapper."""
        return self.index.search

    def as_list(self) -> list[BaseTool]:
        """Both tools, in the order an agent should be given them."""
        return [self.ingest_tool, self.search_tool]


def create_semantic_tools(
    embeddings: Embeddings,
    vector_store: VectorStore,
    store: MemoryStore,
    *,
    search_k: int = 5,
    config: SemanticConfig | None = None,
) -> SemanticTools:
    """Build the ingestion and search tools for one memory tree and one store.

    Pass the result's tools to a factory, or let
    [`create_memory_manager_agent`][deep_memory_agent.agents.create_memory_manager_agent]
    build them for you by handing it `embeddings` and `vector_store` directly::

        from deep_memory_agent import MemoryStore, create_semantic_tools

        semantic = create_semantic_tools(
            OpenAIEmbeddings(model="text-embedding-3-small"),
            InMemoryVectorStore(embeddings),
            MemoryStore(backend),
        )
        semantic.ingest()          # deterministic, no agent involved

    Args:
        embeddings: Embedding model for the chunks and the queries.
        vector_store: Store the chunks are written to and searched in. It must
            upsert on a repeated id, which is what keeps a re-ingest from
            duplicating a chunk.
        store: Store over the memory tree the index covers.
        search_k: Default number of entries a search returns.
        config: Index configuration — chunking, the manifest location, batch
            sizes, an optional server-side filter builder. Defaults to
            [`SemanticConfig`][deep_memory_agent.semantic_index.SemanticConfig].

    Returns:
        The two tools and the index they drive.

    Raises:
        ValueError: If `search_k` is outside `1..25`.
    """
    if not 1 <= search_k <= _MAX_K:
        msg = f"search_k must be positive and at most {_MAX_K}, got {search_k}"
        raise ValueError(msg)

    index = SemanticIndex(
        embeddings, vector_store, store, search_k=search_k, config=config
    )
    snippet_chars = config.snippet_chars if config else _DEFAULT_SNIPPET_CHARS

    def semantic_ingest(
        kind: str | None = None,
        category: str | None = None,
        *,
        only_modified: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """Index the memory tree so it can be searched by meaning."""
        try:
            report = index.ingest(
                kind=MemoryKind(kind) if kind else None,
                category=MemoryCategory(category) if category else None,
                only_modified=only_modified,
            )
        except ValueError as exc:
            return f"Error: {exc}", {}
        return report.summary(), asdict(report)

    def semantic_search(
        query: str,
        k: int = search_k,
        kind: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        min_confidence: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Search the index and render the hits for the model."""
        results = index.search(
            query,
            k=k,
            kind=kind,
            category=category,
            tags=tuple(tags or ()),
            source=source,
            min_confidence=min_confidence,
            created_after=created_after,
            created_before=created_before,
            include_superseded=include_superseded,
        )
        if not results:
            return (
                (
                    "No entry matched by meaning. Try different wording, drop "
                    "the filters, or fall back on `memory_search` — and "
                    "remember the index may not yet cover the most recent "
                    "writes."
                ),
                [],
            )
        return _render(results, snippet_chars), results

    ingest_tool = StructuredTool.from_function(
        func=semantic_ingest,
        name=SEMANTIC_INGEST_TOOL_NAME,
        description=(
            "Index memory entries into the semantic search index, so "
            "`semantic_search` can find them. Run it after `memory_write`, "
            "`memory_update` or `memory_consolidate`. Re-indexing an entry "
            "updates it rather than duplicating it, and unchanged entries are "
            "skipped, so running this is cheap."
        ),
        args_schema=SemanticIngestArgs,
        response_format="content_and_artifact",
    )

    search_tool = StructuredTool.from_function(
        func=semantic_search,
        name=SEMANTIC_SEARCH_TOOL_NAME,
        description=(
            "Search memory by meaning rather than by wording, and get back the "
            "closest excerpts with the id, file and frontmatter of the entry "
            "each came from. Use it when `memory_search` comes up empty because "
            "the question is phrased differently from the entry that answers "
            "it. It returns excerpts, and a long entry can return several: open "
            "what it finds with `memory_get` before relying on it."
        ),
        args_schema=_search_args_schema(search_k),
        response_format="content_and_artifact",
    )

    return SemanticTools(ingest_tool=ingest_tool, search_tool=search_tool, index=index)


def _render(results: list[dict[str, Any]], snippet_chars: int) -> str:
    """Render search hits as the text the model reads.

    The full chunks travel in the tool's artifact; what the model sees is
    trimmed, because a handful of long chunks is how a context window is lost.

    Args:
        results: Hits, best first.
        snippet_chars: Longest excerpt per hit.

    Returns:
        One block per hit, separated by a rule.
    """
    blocks: list[str] = []
    for result in results:
        body = result["text"]
        if len(body) > snippet_chars:
            body = body[:snippet_chars].rstrip() + " […]"
        score = f"{result['score']:.4f}" if result["score"] is not None else "n/a"
        header = (
            f"[{result['rank']}] {result['entry_id']} | {result['path']} | "
            f"{result['category']} | {result['created'][:10]} | "
            f"confidence: {result['confidence']} | "
            f"tags: {', '.join(result['tags']) or '-'} | score: {score}"
        )
        if not result["is_active"]:
            header += " | SUPERSEDED"
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)


def ingest_semantic_index(
    embeddings: Embeddings,
    vector_store: VectorStore,
    *,
    memory_dir: str | Path | None = None,
    backend: BackendProtocol | None = None,
    kind: MemoryKind | None = None,
    category: MemoryCategory | None = None,
    only_modified: bool = True,
    config: SemanticConfig | None = None,
) -> IngestReport:
    """Index the memory tree without an agent in the loop.

    The same code path `semantic_ingest` runs, exposed for a deterministic job —
    a cron entry, a pre-commit hook, the rebuild step of a deployment — where
    nothing should depend on a model deciding to call a tool.

    Args:
        embeddings: Embedding model for the chunks.
        vector_store: Store the chunks are written to.
        memory_dir: Host directory holding the memory tree. Mutually exclusive
            with `backend`; exactly one of the two is required.
        backend: A ready-made backend serving `/memory/`. Mutually exclusive
            with `memory_dir`.
        kind: Restrict the ingest to one memory kind.
        category: Restrict the ingest to one category.
        only_modified: Skip entries that have not changed since the last run.
        config: Index configuration.

    Returns:
        What the run did, as an
        [`IngestReport`][deep_memory_agent.semantic_index.IngestReport].

    Raises:
        ValueError: If both `memory_dir` and `backend` are given, or neither.
    """
    resolved = resolve_backend(memory_dir, backend, for_deep_agent=False)
    store = MemoryStore(resolved)
    store.ensure_tree()
    index = SemanticIndex(embeddings, vector_store, store, config=config)
    return index.ingest(kind=kind, category=category, only_modified=only_modified)
