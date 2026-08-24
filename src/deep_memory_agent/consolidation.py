"""Consolidation: promoting episodes into durable knowledge.

Episodic memory on its own is a log. Without a step that reads it and extracts
what turned out to be stable, the agent has to re-read its whole history to
learn anything from it — expensive and unreliable. Consolidation is that step:
it reads recent episodes, asks a model which of them have hardened into facts,
rules or procedures, and writes those as semantic or procedural entries with
`source: consolidation`.

Episodes are never deleted. Consolidation only ever *adds* durable knowledge and
supersedes semantic entries it contradicts, so the raw history stays auditable.

The function is public and takes a plain backend, so it can be scheduled from
ordinary code — a cron job, a nightly task — without going through a
conversation. The manager agent also exposes it as the `memory_consolidate`
tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from deep_memory_agent.entry import Confidence
from deep_memory_agent.layout import MemoryCategory, MemoryKind
from deep_memory_agent.store import MemoryStore

if TYPE_CHECKING:
    from datetime import datetime

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.language_models import BaseChatModel

    from deep_memory_agent.store import MemoryHit

__all__ = [
    "ConsolidatedItem",
    "ConsolidationProposal",
    "ConsolidationResult",
    "consolidate_memory",
]

_DEFAULT_EPISODE_LIMIT = 100
_BODY_EXCERPT_CHARS = 600

_CONSOLIDATION_PROMPT = """\
You are consolidating an agent's memory. Below are recent episodic entries (raw
history) and the semantic and procedural knowledge currently on file.

Propose only what has *hardened*: something an episode shows to be stable, and
that is not already stated in the durable knowledge below. Propose nothing when
the episodes carry nothing durable — an empty proposal is a correct answer and
far better than restating history as fact.

Rules:
- A recurring situation with a repeatable response becomes a `procedure`; give
  it a `title` and use the sections: When to use, Preconditions, Steps, Tools
  required, Known failures.
- A stable statement about the user or their world becomes a `fact`.
- A constraint on how to behave becomes a `rule`; a stated working preference
  becomes a `preference`.
- If a proposal contradicts an existing entry, set `supersedes` to that entry's
  id instead of writing a second, conflicting statement.
- Use `high` confidence only for something the user stated directly.
- Never copy an episode verbatim. Write the durable conclusion, not the story.

## Recent episodes

{episodes}

## Durable knowledge already on file

{knowledge}
"""


class ConsolidatedItem(BaseModel):
    """One piece of durable knowledge proposed by consolidation."""

    category: Literal["facts", "rules", "preferences", "procedure"] = Field(
        description="Which durable file family this belongs to."
    )
    summary: str = Field(description="One-line description, used in the index.")
    content: str = Field(description="Markdown body of the entry.")
    tags: list[str] = Field(default_factory=list, description="Search labels.")
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium", description="How much this is trusted."
    )
    supersedes: str | None = Field(
        default=None, description="Id of an existing entry this replaces, if any."
    )
    title: str | None = Field(
        default=None,
        description="Procedure title; required when category is procedure.",
    )


class ConsolidationProposal(BaseModel):
    """What the model proposes to promote out of episodic memory."""

    rationale: str = Field(default="", description="Why these items were chosen.")
    items: list[ConsolidatedItem] = Field(
        default_factory=list, description="Entries to write. May be empty."
    )


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Outcome of a consolidation run.

    Attributes:
        entries: The durable entries that were written.
        episodes_considered: How many episodic entries were read.
        rationale: The model's explanation for its choices.
    """

    entries: list[MemoryHit] = field(default_factory=list)
    episodes_considered: int = 0
    rationale: str = ""


def consolidate_memory(
    backend: BackendProtocol,
    model: str | BaseChatModel,
    *,
    since: datetime | None = None,
    limit: int = _DEFAULT_EPISODE_LIMIT,
) -> ConsolidationResult:
    """Promote stable patterns from episodic into semantic and procedural memory.

    Args:
        backend: Backend serving the `/memory/` tree.
        model: Chat model, or a model identifier resolvable by
            `init_chat_model`, used to judge what has hardened.
        since: Only consider episodes created at or after this instant.
        limit: Maximum number of episodes to read.

    Returns:
        The entries written, how many episodes were read, and the model's
        rationale. Writing nothing is a normal outcome.
    """
    store = MemoryStore(backend)
    store.ensure_tree()

    episodes = store.recent_episodes(since=since, limit=limit)
    if not episodes:
        return ConsolidationResult(rationale="no episodes to consolidate")

    knowledge = [
        *store.search(kind=MemoryKind.SEMANTIC, limit=200),
        *store.search(kind=MemoryKind.PROCEDURAL, limit=200),
    ]
    prompt = _CONSOLIDATION_PROMPT.format(
        episodes=_render_hits(episodes),
        knowledge=_render_hits(knowledge) or "(nothing on file yet)",
    )

    chat = init_chat_model(model) if isinstance(model, str) else model
    proposal = chat.with_structured_output(ConsolidationProposal).invoke(prompt)
    if not isinstance(proposal, ConsolidationProposal):
        proposal = ConsolidationProposal.model_validate(proposal)

    written = [_write_item(store, item) for item in proposal.items]
    return ConsolidationResult(
        entries=[hit for hit in written if hit is not None],
        episodes_considered=len(episodes),
        rationale=proposal.rationale,
    )


def _write_item(store: MemoryStore, item: ConsolidatedItem) -> MemoryHit | None:
    """Write one proposed item, skipping anything that cannot be stored."""
    category = MemoryCategory(item.category)
    if category is MemoryCategory.PROCEDURE and not item.title:
        return None

    # A hallucinated id would otherwise retire nothing and leave a dangling
    # `supersedes` pointer in the frontmatter.
    supersedes = item.supersedes or None
    if supersedes and store.get(supersedes) is None:
        supersedes = None

    return store.write(
        category,
        item.content,
        summary=item.summary,
        tags=tuple(item.tags),
        source="consolidation",
        confidence=Confidence(item.confidence),
        supersedes=supersedes,
        title=item.title,
    )


def _render_hits(hits: list[MemoryHit]) -> str:
    """Render entries compactly enough to fit many of them in one prompt."""
    blocks = []
    for hit in hits:
        body = hit.entry.body.strip()
        elided = "…" if len(body) > _BODY_EXCERPT_CHARS else ""
        excerpt = body[:_BODY_EXCERPT_CHARS] + elided
        blocks.append(
            f"- id: {hit.entry.entry_id} | {hit.entry.category.value} | "
            f"{hit.entry.created:%Y-%m-%d} | "
            f"confidence: {hit.entry.confidence.value} | "
            f"tags: {', '.join(hit.entry.tags) or '-'}\n"
            f"  summary: {hit.entry.summary or '-'}\n"
            f"  {excerpt}"
        )
    return "\n".join(blocks)
