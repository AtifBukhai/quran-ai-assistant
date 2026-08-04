# Quran corpus data

## `sample_corpus.json`

A small, runnable subset (Al-Fatihah, Ayat al-Kursi, Al-Ikhlas, Al-Asr, and several verses on
creation, patience, mercy, and forgiveness) so the pipeline works out of the box:

```bash
python -m app.ingest.run --source data/sample_corpus.json
```

Every object matches the canonical schema in `app/models.py` exactly.

## Building the full corpus (6,236 verses)

The full corpus is assembled **offline** from the open [Tanzil Project](https://tanzil.net)
files, distributed under **Creative Commons Attribution 3.0 (CC BY 3.0)**. Attribution to Tanzil
is required when you redistribute the data.

### 1. Download the source files into `data/tanzil/`

| File | What | Where |
|---|---|---|
| `quran-uthmani.txt` | Arabic Uthmani text | https://tanzil.net/download/ (Quran text → Uthmani, "with pipe" / `S|A|text` format) |
| `en.sahih.txt` | Sahih International (English) | https://tanzil.net/trans/ (English → Sahih International) |
| `ur.jalandhry.txt` | Fateh Muhammad Jalandhry (Urdu) | https://tanzil.net/trans/ (Urdu → Jalandhry) |

Each translation download from Tanzil is already in the `S|A|text` pipe format the builder expects.

### 2. Provide structural metadata

Two small JSON files supply the fields Tanzil's text files don't carry inline:

`data/tanzil/surah-metadata.json` — one entry per surah number (1–114):

```json
{
  "1":   {"name_ar": "الفاتحة", "name_en": "Al-Fatihah", "name_ur": "الفاتحہ", "revelation_type": "Makki"},
  "112": {"name_ar": "الإخلاص", "name_en": "Al-Ikhlas",  "name_ur": "الاخلاص", "revelation_type": "Makki"}
}
```

`data/tanzil/ayah-metadata.json` — one entry per verse id (`"S:A"`) with its division numbers:

```json
{
  "1:1":   {"juz": 1,  "hizb": 1,  "ruku": 1,   "page": 1},
  "112:1": {"juz": 30, "hizb": 60, "ruku": 558, "page": 604}
}
```

Juz/hizb/ruku/page can be derived from the Tanzil metadata package (`quran-data.xml` /
`quran-metadata`), or any standard Madani-mushaf mapping. We use the widely-used Hafs/Madani
standard: **114 surahs · 6,236 ayahs · 30 juz · 60 hizb · 558 ruku · 604 pages**.

### 3. Build and ingest

```bash
python -m app.ingest.tanzil --in data/tanzil --out data/full_corpus.json
python -m app.ingest.run --source data/full_corpus.json --full --recreate
```

`--full` asserts the corpus contains exactly 6,236 verses before writing; `--recreate` rebuilds
the Qdrant collection from scratch.

## Attribution

> Quran text and translations © Tanzil Project (https://tanzil.net), licensed CC BY 3.0.
