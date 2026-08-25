# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
