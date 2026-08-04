"""Build the full 6,236-verse corpus from raw Tanzil files.

This is an OFFLINE, build-time utility. It is the only part of the system that touches the
network, and only to fetch the open, redistributable Tanzil data (CC BY 3.0). It never runs on
the answer path.

Inputs (place under data/tanzil/ — see data/README.md for exact URLs):
  - quran-uthmani.txt          Arabic Uthmani text          (format: "S|A|text")
  - en.sahih.txt               Sahih International English   (format: "S|A|text")
  - ur.jalandhry.txt           Jalandhry Urdu                (format: "S|A|text")
  - surah-metadata.json        per-surah names + revelation type (see schema below)
  - ayah-metadata.json         per-ayah juz/hizb/ruku/page   (keyed "S:A")

Output:
  - data/full_corpus.json      canonical schema, ready for `python -m app.ingest.run --full`

Run:
  python -m app.ingest.tanzil --in data/tanzil --out data/full_corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_pipe(path: Path) -> dict[tuple[int, int], str]:
    """Parse a Tanzil 'S|A|text' file into {(surah, ayah): text}."""
    out: dict[tuple[int, int], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        s, a, text = parts
        try:
            out[(int(s), int(a))] = text.strip()
        except ValueError:
            continue
    return out


def build(in_dir: Path, out_path: Path) -> int:
    ar = _parse_pipe(in_dir / "quran-uthmani.txt")
    en = _parse_pipe(in_dir / "en.sahih.txt")
    ur = _parse_pipe(in_dir / "ur.jalandhry.txt")
    surah_meta = json.loads((in_dir / "surah-metadata.json").read_text(encoding="utf-8"))
    ayah_meta = json.loads((in_dir / "ayah-metadata.json").read_text(encoding="utf-8"))

    corpus = []
    for (s, a), text_ar in sorted(ar.items()):
        vid = f"{s}:{a}"
        sm = surah_meta[str(s)]
        am = ayah_meta[vid]
        corpus.append(
            {
                "surah_number": s,
                "ayah_number": a,
                "surah_name": {
                    "ar": sm["name_ar"],
                    "en": sm["name_en"],
                    "ur": sm["name_ur"],
                },
                "text_ar": text_ar,
                "translation_en": en.get((s, a), ""),
                "translation_ur": ur.get((s, a), ""),
                "revelation_type": sm["revelation_type"],  # "Makki" | "Madani"
                "juz": am["juz"],
                "hizb": am["hizb"],
                "ruku": am["ruku"],
                "page": am["page"],
            }
        )

    out_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    return len(corpus)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build full Quran corpus from Tanzil files.")
    p.add_argument("--in", dest="in_dir", type=Path, default=Path("data/tanzil"))
    p.add_argument("--out", dest="out_path", type=Path, default=Path("data/full_corpus.json"))
    args = p.parse_args(argv)
    n = build(args.in_dir, args.out_path)
    print(f"Wrote {n} verses to {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
