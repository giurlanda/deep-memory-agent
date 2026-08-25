"""Generation of the operational-domain corpus.

LongMemEval's evidence sessions are built from a hand-written ontology of 164
personal attributes — demographics, lifestyle, hobbies — because its system under
test is a personal assistant. Two of this benchmark's categories have no data
there at all: LongMemEval never asks an assistant to follow a procedure, and it
has no notion of a mistake that was corrected once and must not come back.

So the ontology is replaced rather than extended, and `ontology.py` is the
equivalent piece of work: the axes of an operational domain — clients, projects,
errors and their corrections, procedures, feedback — from which scenarios are
sampled. It is written once and reused, which is what keeps generated runs
comparable to each other.
"""

from __future__ import annotations

__all__: list[str] = []
