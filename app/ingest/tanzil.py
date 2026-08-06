"""Build the full 6,236-verse corpus from raw Tanzil files.

This is an OFFLINE, build-time utility. It is the only part of the system that touches the
network, and only to fetch the open, redistributable Tanzil data (CC BY 3.0). It never runs on
the answer path.

Inputs (place under data/tanzil/ — see data/README.md for exact URLs):
  - quran-uthmani.txt      Arabic Uthmani text          (format: "S|A|text")
  - en.sahih.txt           Sahih International English   (format: "S|A|text")
  - ur.jalandhry.txt       Jalandhry Urdu                (format: "S|A|text")
  - quran-data.xml         Tanzil structural metadata    (suras + juz/hizb/ruku/page boundaries)
  - surah-metadata.json    per-surah AR/EN/UR names + revelation type (ships in this repo)

Per-ayah juz / hizb / ruku / page are DERIVED from the boundary tables in quran-data.xml
(authoritative), not hand-authored — so they are correct, not guessed.

Output:
  - data/full_corpus.json  canonical schema, ready for `python -m app.ingest.run --full`

Run:
  python -m app.ingest.tanzil --in data/tanzil --out data/full_corpus.json
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
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


def _translator_name(stem: str) -> str:
    """Extract translator name from filename stem (e.g., 'en.yusufali' -> 'Yusuf Ali')."""
    # Remove language prefix
    if "." in stem:
        name_part = stem.split(".", 1)[1]
    else:
        name_part = stem

    # Common Tanzil translator name mappings
    name_map = {
        "yusufali": "Yusuf Ali",
        "pickthall": "Pickthall",
        "shakir": "Shakir",
        "sahih": "Sahih International",
        "maududi": "Maududi",
        "jalandhry": "Jalandhry",
        "kanzuliman": "Kanz-ul-Iman",
        "ahmedali": "Ahmed Ali",
        "arberry": "Arberry",
        "asad": "Asad",
        "daryabadi": "Daryabadi",
        "hilali": "Hilali & Khan",
        "qarai": "Qarai",
        "sarwar": "Sarwar",
        "transliteration": "Transliteration",
    }

    return name_map.get(name_part.lower(), name_part.title())


def _assign_from_boundaries(
    order: list[tuple[int, int]],
    boundaries: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Assign each ayah the 1-based unit number it belongs to.

    ``order`` is every (sura, aya) in mushaf order. ``boundaries`` is the list of unit-start
    markers, also in mushaf order: the k-th marker (0-based) begins unit k+1. Every ayah from
    one marker up to (but not including) the next belongs to that marker's unit.
    """
    pos = {va: i for i, va in enumerate(order)}
    starts = sorted(pos[b] for b in boundaries)  # global start index of each unit
    assign: dict[tuple[int, int], int] = {}
    si = 0
    for gi, va in enumerate(order):
        # Advance while the next boundary has started at or before this ayah.
        while si + 1 < len(starts) and gi >= starts[si + 1]:
            si += 1
        assign[va] = si + 1  # 1-based unit number
    return assign


def _boundaries(root: ET.Element, container: str, child: str) -> list[tuple[int, int]]:
    """Read <container><child sura= aya=/></container> markers into sorted (sura, aya) tuples."""
    node = root.find(container)
    if node is None:
        return []
    marks = [
        (int(c.attrib["sura"]), int(c.attrib["aya"]))
        for c in node.findall(child)
    ]
    return marks


def build(in_dir: Path, out_path: Path) -> int:
    # Load base translations (required)
    ar = _parse_pipe(in_dir / "quran-uthmani.txt")
    en = _parse_pipe(in_dir / "en.sahih.txt")
    ur = _parse_pipe(in_dir / "ur.jalandhry.txt")
    surah_meta = json.loads((in_dir / "surah-metadata.json").read_text(encoding="utf-8"))

    # Auto-detect additional English translations (en.*.txt except en.sahih.txt)
    extra_en = []
    extra_en_names = []
    for f in sorted(in_dir.glob("en.*.txt")):
        if f.name != "en.sahih.txt":
            extra_en.append(_parse_pipe(f))
            extra_en_names.append(_translator_name(f.stem))
            print(f"  + Found extra English translation: {f.name}")

    # Auto-detect additional Urdu translations (ur.*.txt except ur.jalandhry.txt)
    extra_ur = []
    extra_ur_names = []
    for f in sorted(in_dir.glob("ur.*.txt")):
        if f.name != "ur.jalandhry.txt":
            extra_ur.append(_parse_pipe(f))
            extra_ur_names.append(_translator_name(f.stem))
            print(f"  + Found extra Urdu translation: {f.name}")

    root = ET.parse(in_dir / "quran-data.xml").getroot()
    order = sorted(ar.keys())  # (sura, aya) in mushaf order

    juz_of = _assign_from_boundaries(order, _boundaries(root, "juzs", "juz"))
    # Tanzil stores 240 quarters (rub' al-hizb); hizb = ceil(quarter / 4).
    quarter_of = _assign_from_boundaries(order, _boundaries(root, "hizbs", "quarter"))
    ruku_of = _assign_from_boundaries(order, _boundaries(root, "rukus", "ruku"))
    page_of = _assign_from_boundaries(order, _boundaries(root, "pages", "page"))

    corpus = []
    for (s, a) in order:
        vid = f"{s}:{a}"
        sm = surah_meta[str(s)]
        quarter = quarter_of[(s, a)]
        hizb = (quarter + 3) // 4  # ceil(quarter/4)

        verse = {
            "surah_number": s,
            "ayah_number": a,
            "surah_name": {
                "ar": sm["name_ar"],
                "en": sm["name_en"],
                "ur": sm["name_ur"],
            },
            "text_ar": ar[(s, a)],
            "translation_en": en.get((s, a), ""),
            "translation_ur": ur.get((s, a), ""),
            "revelation_type": sm["revelation_type"],  # "Makki" | "Madani"
            "juz": juz_of[(s, a)],
            "hizb": hizb,
            "ruku": ruku_of[(s, a)],
            "page": page_of[(s, a)],
        }

        # Add extra English translations as translation_en_2, translation_en_3, etc.
        for i, extra_dict in enumerate(extra_en, start=2):
            verse[f"translation_en_{i}"] = extra_dict.get((s, a), "")

        # Add extra Urdu translations as translation_ur_2, translation_ur_3, etc.
        for i, extra_dict in enumerate(extra_ur, start=2):
            verse[f"translation_ur_{i}"] = extra_dict.get((s, a), "")

        corpus.append(verse)

    out_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=0), encoding="utf-8"
    )

    # Write translation manifest for the UI
    manifest = {
        "english": [
            {"key": "1", "label": "Sahih International (primary)", "filename": "en.sahih.txt"}
        ],
        "urdu": [
            {"key": "1", "label": "Jalandhry (primary)", "filename": "ur.jalandhry.txt"}
        ],
    }

    for i, name in enumerate(extra_en_names, start=2):
        manifest["english"].append({"key": str(i), "label": name, "filename": f"translation_en_{i}"})

    for i, name in enumerate(extra_ur_names, start=2):
        manifest["urdu"].append({"key": str(i), "label": name, "filename": f"translation_ur_{i}"})

    manifest_path = out_path.parent / "translations.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote translation manifest to {manifest_path}")

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
