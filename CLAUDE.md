# deep-memory-agent

Factory for building LangChain deepagents equipped with episodic, semantic, and procedural memory.

## Layout

- `src/deep_memory_agent/` — library source (src layout; nothing importable from the repo root).
- `tests/` — pytest suite, mirrors the package structure.
- `examples/` — small runnable scripts demonstrating usage; not part of the published package.
- `docs/` — MkDocs + Material source; built with mkdocstrings from docstrings in `src/`.
- `CHANGELOG.md` — Keep a Changelog format; add entries under `[Unreleased]` as you go.

## Packaging & versioning

- Managed with [uv](https://docs.astral.sh/uv/); `pyproject.toml` uses the `hatchling` build backend.
- The version lives in `src/deep_memory_agent/__init__.py` (`__version__`) and `[tool.hatch.version]`
  reads it from there — bump that one place, nothing else, when cutting a release.
- Follow [Semantic Versioning](https://semver.org/): patch for fixes, minor for backwards-compatible
  features, major for breaking changes.
- Requires Python >=3.12.

## Tests

- Run with `uv run pytest`. Add a test alongside every behavior change — this project does not
  merge untested changes to `src/`.
- Keep fixtures in `tests/conftest.py`.

## Lint & format

- `uv run ruff check .` and `uv run ruff format .` before committing. Ruff selects `ALL` rules by
  default (see `[tool.ruff.lint]` in `pyproject.toml`); add narrowly-scoped ignores with a comment
  explaining why, rather than disabling a rule project-wide.

## Documentation

- Built with MkDocs + Material, source in `docs/`, config in `mkdocs.yml`.
- The API reference page uses `mkdocstrings` to pull docstrings straight from `src/deep_memory_agent/`
  (Google docstring convention) — write real docstrings, they are the reference docs.
- Preview locally with `uv run mkdocs serve`; `uv run mkdocs build --strict` is what CI runs, so a
  broken cross-reference or nav entry fails the build the same way locally as in CI.

## CI/CD

- `.github/workflows/ci.yml` runs on every push to `main` and every PR: ruff lint + format check,
  `pytest` across the supported Python matrix, a `uv build` sanity check, and an mkdocs --strict build.
- `.github/workflows/publish.yml` runs when a `v*.*.*` tag is pushed: it verifies the tag matches
  `__version__` in `src/deep_memory_agent/__init__.py`, builds, and publishes to PyPI via trusted
  publishing (no stored token — the `pypi` GitHub environment must be configured with a PyPI
  Trusted Publisher). To release: bump `__version__`, update `CHANGELOG.md`, merge, then
  `git tag vX.Y.Z && git push --tags`.

## Workflow expectations

- New features and bug fixes need a matching test and, if they touch public API, a docs update.
- Update `CHANGELOG.md` under `[Unreleased]` for every user-facing change.
- Don't hand-edit the version number anywhere except `src/deep_memory_agent/__init__.py`.
