"""Command-line query interface for the grounded Quran assistant.

Runs the full RAG pipeline (the same one behind ``POST /v1/ask``) against the indexed
corpus and prints the answer plus its cited evidence — no server required.

Usage:
    python -m app.ask "What does the Quran say about patience?"
    python -m app.ask "Recite verse 112:1" --mode ar_en
    python -m app.ask --json "who created the heavens"

Options:
    --mode {ar,ar_en,ar_ur,ar_en_ur}   Which translations to show (default: ar_en_ur).
    --lang {ar,en,ur}                  Force query language (default: auto-detect).
    --json                             Emit the raw AskResponse as JSON instead of prose.

Exit code is 0 when a grounded answer is returned, 1 on any refusal (out-of-scope,
low-confidence, or insufficient evidence) — so the command is scriptable.
"""

from __future__ import annotations

import argparse
import sys

from .models import AskResponse
from .orchestrator import Orchestrator

_MODES: tuple[str, ...] = ("ar", "ar_en", "ar_ur", "ar_en_ur")


def _render(resp: AskResponse) -> str:
    lines: list[str] = []
    if resp.is_refusal:
        lines.append(f"✗ {resp.status.upper()}")
        lines.append(resp.answer)
    else:
        lines.append("✓ ANSWER")
        lines.append(resp.answer)
        if resp.citations:
            lines.append("")
            lines.append(f"Citations: {', '.join(resp.citations)}")

    if resp.evidence:
        lines.append("")
        lines.append("Evidence:")
        for ev in resp.evidence:
            names = ev.surah_name
            lines.append(
                f"  [{ev.verse_id}] {names.en} "
                f"({ev.revelation_type.value}, juz {ev.juz})  score={ev.score:.3f}"
            )
            lines.append(f"    AR: {ev.text_ar}")
            if ev.translation_en:
                lines.append(f"    EN: {ev.translation_en}")
            if ev.translation_ur:
                lines.append(f"    UR: {ev.translation_ur}")

    lines.append("")
    lines.append(f"[intent={resp.mode or '-'} lang={resp.language} trace={resp.trace_id}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ask",
        description="Ask the strictly-grounded Quran assistant a question.",
    )
    parser.add_argument("query", help="The question or verse reference to ask.")
    parser.add_argument(
        "--mode",
        choices=_MODES,
        default="ar_en_ur",
        help="Which translations to display (Arabic is always shown).",
    )
    parser.add_argument(
        "--lang",
        choices=("ar", "en", "ur"),
        default=None,
        help="Force the query language instead of auto-detecting.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw AskResponse as JSON.",
    )
    args = parser.parse_args(argv)

    orch = Orchestrator()
    resp = orch.answer(args.query, mode=args.mode, lang=args.lang)  # type: ignore[arg-type]

    if args.json:
        print(resp.model_dump_json(indent=2))
    else:
        print(_render(resp))

    return 1 if resp.is_refusal else 0


if __name__ == "__main__":
    raise SystemExit(main())
