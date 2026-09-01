"""Building the embedding index a case is searched through.

The index is built once, after the last session has been replayed and before the
question is asked — the same place the `final` consolidation pass runs, and for
the same reason. In production `semantic_ingest` is the manager's own tool and
the index trails the files by however long the agent takes to call it; measuring
that lag here would score the run on when the last write landed rather than on
what the tree holds. So the harness drives the ingest, exactly like
consolidation, and both arms answer over a tree that is finished.

`ingest_semantic_index` is the shipped code path with no model in the middle,
which is what makes the index deterministic: the same tree produces the same
chunks on every run, and a resumed case can rebuild it without re-ingesting a
single session.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from dma_bench.schema import SemanticIndexRecord

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

__all__ = ["index_case"]


def index_case(
    memory_dir: Path,
    embeddings: Embeddings,
    vector_store: VectorStore,
) -> SemanticIndexRecord:
    """Index a case's finished memory tree into its vector store.

    Args:
        memory_dir: Directory holding the memory tree.
        embeddings: The embedding model.
        vector_store: The store this case's chunks go to. It must be the case's
            own: two cases sharing a store would let one answer out of the
            other's memory.

    Returns:
        What the pass covered and cost. A failure is recorded rather than
        raised, so a case that cannot be indexed still keeps its lexical arm.
    """
    from deep_memory_agent import ingest_semantic_index

    started = time.monotonic()
    try:
        report = ingest_semantic_index(
            embeddings, vector_store, memory_dir=memory_dir, only_modified=True
        )
    except Exception as exc:
        return SemanticIndexRecord(
            error=repr(exc), duration_s=round(time.monotonic() - started, 2)
        )
    return SemanticIndexRecord(
        added=report.added,
        updated=report.updated,
        deleted=report.deleted,
        unchanged=report.unchanged,
        chunks=report.chunks,
        deleted_chunks=report.deleted_chunks,
        entry_errors=list(report.errors),
        duration_s=round(time.monotonic() - started, 2),
    )
