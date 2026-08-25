"""The three graders, one per stage of the pipeline.

LongMemEval grades two stages, retrieval and generation, because it has two. A
system that writes its own memory has three, and the extra one comes first: a
fact can be missing from the answer because consolidation never extracted it,
because search never surfaced it, or because the answer misread it. Those are
three different bugs in three different prompts, and a single accuracy number
cannot tell them apart.

Every judge returns structured output rather than a parsed "yes"/"no", and keeps
the reasoning that produced it, so a surprising score can be audited instead of
believed.
"""

from __future__ import annotations

__all__: list[str] = []
