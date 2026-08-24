# deep-memory-agent

Factory for building LangChain deepagents equipped with episodic, semantic, and procedural memory.

## Installation

```bash
pip install deep-memory-agent
```

## Quickstart

```python
from deep_memory_agent import __version__

print(__version__)
```

See [examples/basic_usage.py](examples/basic_usage.py) for a runnable example, or see the [docs](https://giurlanda.github.io/deep-memory-agent/).

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run ruff format .
```

## License

MIT — see [LICENSE](LICENSE).
