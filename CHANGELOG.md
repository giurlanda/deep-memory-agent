# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-01

### Added

- Semantic search over the memory tree: an embedding index alongside the lexical
  one, covering all seven categories. The markdown files stay the single source
  of truth — the index is derived data, rebuildable from `/memory/` at any time
  and deletable without losing a fact. Enabled by passing `embeddings` and
  `vector_store` to either agent factory; needs the new optional `semantic`
  extra. ([#10])
- `semantic_ingest` and `semantic_search` tools. The manager agent gets both,
  since it is the single writer and so the only agent that can keep a derived
  index in step; the recall agent gets only the search, because withholding the
  ingest tool is the only way to keep it read-only over a store that
  `READ_ONLY_MEMORY_PERMISSIONS` cannot guard.
- `ingest_semantic_index`, the same ingest without a model in the loop, for a
  cron entry, a pre-commit hook or a deployment step. Re-running it with nothing
  changed leaves the vector store as it was and says so.
- `SemanticIndex`, `SemanticConfig`, `ChunkingConfig`, `IngestReport`,
  `SemanticTools`, `chunk_entry` and `create_semantic_tools` on the public API,
  plus a `Semantic search` page in the docs.
- `memory_get`, a recall tool that returns one entry in full given its id.
  `memory_search` truncates long bodies and `memory_read` hands back every entry
  in a file, so there was no way to open exactly the entry a hit pointed at —
  a gap the semantic index makes routine.
- `for_deep_agent` on `resolve_backend`, so a standalone caller can resolve a
  `memory_dir` without landing on a `StateBackend` that only answers inside a
  graph execution.
- Two runnable examples for the semantic index, `examples/build_semantic_memory.py`
  and `examples/semantic_memory.py`, against a hybrid Qdrant collection (dense +
  BM25) with embeddings served locally over an OpenAI-compatible endpoint. The
  second runs one question through `memory_search` and `semantic_search` side by
  side with no model in the loop, so the gap between them is visible rather than
  asserted.
- An `examples` dependency group holding what those scripts need. Like
  `benchmark`, it is outside `uv sync --all-extras`, so CI and the published
  wheel are unaffected.

### Changed

- `create_memory_search_agent` and `create_memory_manager_agent` take
  `embeddings`, `vector_store`, `search_k` and `semantic_config`. All default to
  off; without them the agents behave exactly as before. When semantic search is
  active, the built-in prompts gain a section on it — a `system_prompt` of your
  own still replaces the prompt whole, that section included.

[#10]: https://github.com/giurlanda/deep-memory-agent/issues/10

## [0.1.4] - 2026-08-31

### Added

- `--cases-per-category` on the benchmark corpus generator, and the matching
  `cases_per_category` argument on `generate_corpus`, overriding the case count
  the chosen `--config` fixes while leaving the sessions per case and the
  timeline span alone — so a trial corpus can use the `large` timeline without
  paying for a `large` run.
- Checkpointing and resume in the benchmark corpus generator. Every finished
  case is written to `--out` immediately, and a run pointed at an existing
  corpus keeps what is there and generates only the cases missing to reach the
  requested count — so an interrupted run is restarted with the same command,
  and an existing corpus is grown by asking for more cases. `generate_corpus`
  takes the destination as `out=`, and `load_corpus` reads a corpus back.

## [0.1.3] - 2026-08-25

### Added

- A LongMemEval-style benchmark for the two memory agents, under `benchmark/`,
  driven by `benchmark/run_benchmark.ipynb`. It replays episodes through the
  real manager agent one session at a time, so extraction, supersession and
  consolidation are part of what is measured — not a retriever over pre-loaded
  facts — and decomposes failures across three stages (consolidation ×
  retrieval × answer) rather than LongMemEval's two. Six question categories
  come from the published LongMemEval datasets; `supersede-integrity` re-judges
  the knowledge-update cases under a stricter prompt that fails a retired value
  presented as current; `procedural-retrieval` and `non-repetition` come from a
  generated corpus over an operational-domain ontology. ([#7])
- A `benchmark` dependency group holding what the harness needs on top of the
  package. It is not installed by `uv sync`, so CI and the published wheel are
  unaffected.
- `docs/benchmark.md`, and a `Benchmark` entry in the MkDocs nav.

### Changed

- `.gitignore` no longer excludes the whole `benchmark/` directory: the harness
  is tracked, while the multi-gigabyte LongMemEval datasets, the paper PDF and
  the run outputs stay out.

[#7]: https://github.com/giurlanda/deep-memory-agent/issues/7

## [0.1.2] - 2026-08-25

### Added

- `build_memory_backend(..., for_deep_agent=False)` builds a backend usable
  outside a deep agent, serving non-memory paths from an empty scratch
  directory instead of LangGraph thread state. The default stays `True`, so
  agent wiring is unchanged.
- CI now deploys the built MkDocs site to GitHub Pages on every push to `main`,
  via a `deploy-docs` job gated on `lint`, `test`, `build`, and `docs` passing.

### Fixed

- `consolidate_memory` (and `MemoryStore` generally) no longer fails with
  `RuntimeError: StateBackend must be used inside a LangGraph graph execution`
  when handed a backend from `build_memory_backend`. Listing memory files
  issues an unscoped `glob`, which `CompositeBackend` fans out to the default
  backend as well as the `/memory/` route — reaching the graph-only
  `StateBackend` even though no memory path lives there. ([#5])

## [0.1.1] - 2026-08-24

### Added

- `create_memory_search_agent` and `create_memory_manager_agent`: two factories
  over one memory tree. The search agent answers from memory and is denied every
  write on `/memory/` at the backend level; the manager agent is the single
  writer.
- File-based memory tree at the virtual path `/memory/`, split into episodic,
  semantic and procedural memory, with a router `index.md` per directory and
  episodic files sharded by month.
- YAML frontmatter on every entry (`id`, `created`, `type`, `category`,
  `source`, `confidence`, `tags`, `supersedes`, `superseded_by`), so provenance
  travels with the content and a corrected fact retires the one it replaces.
- Memory tools built over the deepagents backend — `memory_index`,
  `memory_search`, `memory_read`, `memory_write`, `memory_update`,
  `memory_consolidate` — none of which touch the host filesystem directly.
- `consolidate_memory`, a public function that promotes stable patterns from
  episodic memory into facts, rules or procedures, schedulable from plain code
  and also exposed to the manager agent as a tool.
- `MemoryStore`, `build_memory_backend`, `resolve_backend` and
  `ensure_memory_tree` as the supporting public surface.

[Unreleased]: https://github.com/giurlanda/deep-memory-agent/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/giurlanda/deep-memory-agent/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/giurlanda/deep-memory-agent/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/giurlanda/deep-memory-agent/releases/tag/v0.1.0
[#5]: https://github.com/giurlanda/deep-memory-agent/issues/5
