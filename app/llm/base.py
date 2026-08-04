"""The generation interface.

A ``GroundedLLM`` receives the strict system prompt and a user prompt that already contains the
retrieved verses. It returns raw answer text (which is then validated by ``app.validator``).
The LLM is treated as a constrained rewriter; it is never trusted and never given tools.
"""

from __future__ import annotations

from typing import Protocol


class GroundedLLM(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's answer text (may contain [S:A] citations or a refusal)."""
        ...
