# Adding Multiple Translations

The assistant now supports **pluggable additional translations** to improve keyword coverage and retrieval accuracy.

## How It Works

The system automatically detects extra translation files in `data/tanzil/` during corpus building:

- **English translations**: Any `en.*.txt` file (except `en.sahih.txt`)
- **Urdu translations**: Any `ur.*.txt` file (except `ur.jalandhry.txt`)

Extra translations are indexed alongside the base translations, and all retrieval methods (keyword search, lexical overlap, semantic search, word concordance) will search across them.

## Setup Steps

### 1. Download Additional Translations

Visit [tanzil.net/trans/](http://tanzil.net/trans/) and download additional translation files in Tanzil pipe format (`S|A|text`).

**Recommended English translations:**
- `en.yusufali.txt` — Yusuf Ali
- `en.pickthall.txt` — Pickthall
- `en.shakir.txt` — Shakir

**Recommended Urdu translations:**
- `ur.maududi.txt` — Maududi
- `ur.kanzuliman.txt` — Kanz-ul-Iman

Place downloaded files directly into `data/tanzil/`.

### 2. Rebuild the Corpus

```bash
python -m app.ingest.tanzil --in data/tanzil --out data/full_corpus.json
```

The script will detect and report extra translations:
```
  + Found extra English translation: en.yusufali.txt
  + Found extra Urdu translation: ur.maududi.txt
Wrote 6236 verses to data/full_corpus.json
```

### 3. Re-ingest into Qdrant

```bash
python -m app.ingest.run --source data/full_corpus.json --full --recreate
```

The `--recreate` flag drops and recreates the collection with the new translation fields.

## What Gets Indexed

Each extra translation is stored as:
- **Raw field**: `translation_en_2`, `translation_en_3`, ... or `translation_ur_2`, `translation_ur_3`, ...
- **Normalized field**: `translation_en_2_lower`, `translation_ur_2_normalized`, etc. (used for matching)

All normalized fields are searched during keyword/lexical retrieval, and all raw translations are included in the dense embedding vector.

## Example

**Before** (only `en.sahih.txt`):
- Query: "Isa" (Jesus in Arabic transliteration)
- Result: Poor recall if "Isa" appears in Yusuf Ali but not in Sahih International

**After** (with `en.yusufali.txt` added):
- Query: "Isa"
- Result: Finds verses where Yusuf Ali uses "Isa" even if Sahih International uses "Jesus"

This directly addresses the issue where common terms like "اسوہ حسنہ" or "Isa" were missed because the single base translation didn't use those exact words.

## Notes

- The base three translations (`quran-uthmani.txt`, `en.sahih.txt`, `ur.jalandhry.txt`) remain **required**.
- Extra translations are **optional** — the system works without them.
- Re-ingesting with `--recreate` is **necessary** after adding translations; the old collection won't have the new fields.
- Supported backends (hash or sentence-transformers) both benefit from extra translations in lexical/keyword paths. Semantic search benefits more with sentence-transformers.
