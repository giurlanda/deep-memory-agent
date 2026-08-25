"""A LongMemEval-style benchmark for `deep_memory_agent`.

The methodology comes from LongMemEval (ICLR 2025): a taxonomy of question
types, one LLM judge prompt per type, retrieval scored separately from
generation, and failures decomposed by stage. What changes is the system under
test. LongMemEval evaluates a retriever over a frozen chat log; here the
episodes are replayed **through the real manager agent**, one session at a
time, so `memory_write`, supersession and `memory_consolidate` are all part of
what is being measured.

That extra stage is why the error decomposition is three-way — consolidation ×
retrieval × answer, eight cells instead of four. A wrong answer means something
different depending on where it broke: a fact the consolidation step never
extracted, a fact in memory the search never surfaced, or a fact that was
surfaced and then misread.

The entry point is `benchmark/run_benchmark.ipynb`; everything here is the
library it drives.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
