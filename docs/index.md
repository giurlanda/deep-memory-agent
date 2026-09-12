# deep-memory-agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/giurlanda/deep-memory-agent?sort=semver&label=version)](https://github.com/giurlanda/deep-memory-agent/tags)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

Factory for building LangChain deepagents equipped with episodic, semantic, and
procedural memory.

Memory is a small wiki of plain markdown files that the agent maintains itself:
it records what happened, supersedes what is no longer true, and consolidates
recurring patterns into durable knowledge. The tree lives at the virtual path
`/memory/`, served by a deepagents backend — no tool in this package touches the
host filesystem directly.

The package ships **two** agents over that tree, because recall and curation pull
in opposite directions:

| Factory | Can write? | Use it for |
| --- | --- | --- |
| [`create_memory_search_agent`][deep_memory_agent.create_memory_search_agent] | No, denied at the backend | Answering from memory |
| [`create_memory_manager_agent`][deep_memory_agent.create_memory_manager_agent] | Yes, and it is the only writer | Recording, superseding, consolidating |

Both agents can also be given an embedding model and a vector store, which adds
an index that finds an entry whose wording does not match the question — see
[Semantic search](semantic-search.md).

Start with [Getting started](getting-started.md), read
[Memory layout](memory-layout.md) for the file format and the rules that govern
it, or jump to the [API reference](api.md).
