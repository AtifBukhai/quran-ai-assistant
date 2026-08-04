"""Offline, deterministic grounded LLM.

``EchoGroundedLLM`` does NOT call any network or model. It performs a tiny, fully deterministic
"generation" that is nonetheless strictly grounded: it composes an answer purely by quoting the
provided context verses and citing their ids. This lets the entire pipeline — routing,
retrieval, confidence gating, validation, response assembly — run and be unit-tested with no API
key and no internet, which is exactly what the grounding guarantees require us to be able to
verify.

Because it only ever emits verse ids that appear in the context, it passes the same validator
the cloud model's output must pass. If the context is empty it emits the exact refusal string.
"""

from __future__ import annotations

import re

from ..grounding import REFUSAL_INSUFFICIENT

_CONTEXT_VERSE_RE = re.compile(r"^\[(\d{1,3}:\d{1,3})\] \(([^)]*)\)", re.MULTILINE)
_EN_LINE_RE = re.compile(r"^\s*English:\s*(.+)$", re.MULTILINE)


class EchoGroundedLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        verse_ids = _CONTEXT_VERSE_RE.findall(user_prompt)
        english = _EN_LINE_RE.findall(user_prompt)
        if not verse_ids:
            return REFUSAL_INSUFFICIENT

        # Compose a concise grounded answer strictly from the context's own English lines.
        # (Arabic in the final response is re-rendered from the canonical corpus by the
        # assembler, so we never author verse text here.)
        # verse_ids is [(vid, surah_name), ...]; extract just the id strings.
        ids = [vid for vid, _name in verse_ids]
        parts = []
        for vid, en in zip(ids, english):
            snippet = en.strip().rstrip(".")
            parts.append(f"{snippet} [{vid}].")
        # Keep it short: at most the first two supporting verses in the summary sentence.
        answer = " ".join(parts[:2])
        # Ensure every retrieved verse is still cited (spec: cite every verse used).
        cited = set(re.findall(r"\[(\d{1,3}:\d{1,3})\]", answer))
        trailing = [vid for vid in ids if vid not in cited]
        if trailing:
            answer += " " + " ".join(f"[{vid}]" for vid in trailing)
        return answer.strip()
