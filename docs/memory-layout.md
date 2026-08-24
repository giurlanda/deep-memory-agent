# Memory layout

Memory is a wiki of plain markdown files under the virtual path `/memory/`. The
format is deliberately boring: readable by a human, diffable by git, and cheap
to index later without changing the source of truth.

## The tree

```
/memory/
├── index.md                      router for the whole tree
├── preferences.md                how the user wants the agent to behave
├── episodic_memory/              what happened
│   ├── index.md
│   ├── events/YYYY-MM.md         sessions and decisions, sharded by month
│   ├── feedbacks/YYYY-MM.md      corrections and judgements from the user
│   └── errors/YYYY-MM.md         mistakes made, so they are not repeated
├── semantic_memory/              what is true
│   ├── index.md
│   ├── facts.md
│   └── rules.md
└── procedural_memory/            how things are done
    ├── index.md
    └── <slug>.md                 one file per procedure
```

The split follows the standard agent-memory taxonomy: **episodic** for what
happened, **semantic** for what is true, **procedural** for how things are done.
Routing retrieval by kind is what keeps the agent from dumping the whole tree
into its context to answer one question.

Preferences sit at the root rather than under `semantic_memory/` because they cut
across kinds, but they are treated as semantic memory: a changed preference
supersedes the previous one.

[`ensure_memory_tree`][deep_memory_agent.ensure_memory_tree] creates this tree
through the backend and never overwrites an existing file, so it is safe to call
on every agent construction. Episodic shards are the exception: they are named
after the month they cover and appear the first time something is written to
them.

## Entries and frontmatter

A memory file is an append-only document made of entries, each carrying YAML
frontmatter:

```markdown
---
id: mem_2026-08-24_9f3a1c
created: 2026-08-24T10:15:00+00:00
type: semantic
category: facts
source: user_message
confidence: high
tags: [pricing, acme]
supersedes: mem_2026-06-01_4b2e77
summary: ACME moved to the Enterprise plan
---

ACME switched from the Team plan to Enterprise on 2026-08-24.
```

| Field | Meaning |
| --- | --- |
| `id` | Unique, dated identifier. Cite it when relying on an entry. |
| `created` | Timezone-aware creation instant. |
| `type` | `episodic`, `semantic` or `procedural`, derived from `category`. |
| `category` | `events`, `feedbacks`, `errors`, `facts`, `rules`, `preferences` or `procedure`. |
| `source` | Where it came from: `user_message`, `user_feedback`, `tool_result`, `consolidation`, `agent`. |
| `confidence` | `low`, `medium` or `high`. |
| `tags` | Labels used to narrow searches. |
| `supersedes` | Id of the entry this one replaces. |
| `superseded_by` | Id of the entry that replaced this one. |

Provenance is what makes memory debuggable. Without it, "why does the agent
believe X?" has no answer.

An entry body must not contain a line that is exactly `---`: that is how entries
are delimited, and [`render_entry`][deep_memory_agent.entry.render_entry] rejects
it rather than writing a file it could not read back.

## Supersession

Semantic memory is *corrected*, not accumulated. When a fact changes, the new
entry records `supersedes: <old id>` and the old entry is marked
`superseded_by: <new id>` in place. The old entry stays on disk as history but
drops out of search results unless `include_superseded` is set.

Episodic memory works the other way: it is never rewritten. An episode was true
at the moment it happened, and stays that way.

## Indexes are routers

Every directory owns an `index.md` whose only job is to let an agent decide what
to load:

```markdown
| File | Description | Tags | Updated |
| --- | --- | --- | --- |
| events/2026-08.md | Session events for August 2026 | acme, pricing | 2026-08-24 |
```

Indexes never hold memory content. Descriptions come from the `summary` of the
most recent write; tags accumulate across writes so the index keeps working as a
lookup table. Any hand-written preamble above the table survives updates.

## Sharding

Episodic categories are sharded by month — `events/2026-08.md` — so no single
file grows past what fits in a context window. Semantic files are not sharded:
supersession keeps them from growing without bound, since a corrected fact
replaces rather than adds.

## Procedures

Each procedure is one file with a fixed shape, so the agent knows where to look
under time pressure:

```markdown
## When to use
## Preconditions
## Steps
## Tools required
## Known failures
```

`Known failures` is where episodic memory feeds back into procedural memory: a
mistake that keeps recurring in `errors/` belongs there, next to the steps it
would otherwise derail. See
[`PROCEDURE_TEMPLATE`][deep_memory_agent.PROCEDURE_TEMPLATE].

## Consolidation

Consolidation is the bridge between kinds. It reads recent episodes, asks a model
which have hardened into stable knowledge, and writes those as semantic or
procedural entries with `source: consolidation`, superseding whatever they
contradict. Episodes themselves are never deleted, so the raw history stays
auditable.

Without this step, episodic memory is only a log: the agent would have to re-read
its whole history to learn anything from it.

## Retrieval, and what comes after it

Search is lexical: a case-insensitive substring test over each entry's summary,
body and tags, filtered by kind, category and tags. That is enough for the volume
a file-based memory is meant to hold, and it has no infrastructure cost.

Its limit is real — synonyms and paraphrases are missed. The structured
frontmatter is what makes the upgrade path cheap: a BM25 or vector index can be
built over these files later without changing them, because every entry already
carries the identity, timestamps and relationships such an index needs.
