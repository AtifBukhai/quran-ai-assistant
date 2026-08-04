"""Cloud generation adapter (Anthropic Claude), used as a *constrained rewriter* only.

Strict grounding is enforced structurally:
* temperature = 0 (determinism)
* the system prompt is the strict-grounding contract
* the user prompt contains ONLY retrieved verses + the question
* no tools, no web, no memory are passed
* the model's output is still validated downstream by app.validator — the model is never trusted

Requires ``pip install -e ".[cloud]"`` and ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations


class AnthropicGroundedLLM:
    def __init__(self, model: str) -> None:
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate text blocks.
        return "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        ).strip()
