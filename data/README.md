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
| `quran-data.xml` | Structural metadata (juz/hizb/ruku/page boundaries) | https://tanzil.net/docs/quran-metadata (or the `quran-metadata` package) |

Each translation download from Tanzil is already in the `S|A|text` pipe format the builder
expects. `quran-data.xml` is Tanzil's canonical metadata file: it carries the authoritative
`<juzs>`, `<hizbs>` (240 quarters), `<rukus>`, and `<pages>` boundary tables.

### 2. Structural metadata — already in the repo

`data/tanzil/surah-metadata.json` **ships in this repo** — one entry per surah number (1–114)
with Arabic/English/Urdu names and the Makki/Madani revelation type. You don't need to author
anything.

Per-ayah **juz / hizb / ruku / page are derived** by the builder from the boundary tables in
`quran-data.xml` (authoritative), not hand-authored — so they are correct rather than guessed.
The builder computes `hizb = ceil(quarter / 4)` from Tanzil's 240 quarters. There is **no**
`ayah-metadata.json` to maintain. The result matches the Hafs/Madani standard:
**114 surahs · 6,236 ayahs · 30 juz · 60 hizb · 558 ruku · 604 pages**.

### 3. Build and ingest

```bash
# Merge Tanzil files + derive structure -> data/full_corpus.json
python -m app.ingest.tanzil --in data/tanzil --out data/full_corpus.json

# Validate (must be exactly 6,236 verses) and index into Qdrant
python -m app.ingest.run --source data/full_corpus.json --full --recreate
```

`--full` asserts the corpus contains exactly 6,236 verses before indexing; `--recreate` rebuilds
the Qdrant collection from scratch.

## Attribution

> Quran text and translations © Tanzil Project (https://tanzil.net), licensed CC BY 3.0.
