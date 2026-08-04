"""Corpus schema & invariant tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    TOTAL_SURAHS,
    Verse,
    normalize_arabic,
    parse_verse_id,
)


def test_verse_derived_fields(sample_verses):
    v = next(x for x in sample_verses if x.verse_id == "112:1")
    assert v.surah_number == 112
    assert v.ayah_number == 1
    assert v.word_count == len(v.text_ar.split())
    assert v.char_count == len(v.text_ar.replace(" ", ""))
    assert len(v.checksum) == 64  # sha256 hex
    assert "[AR]" in v.embedding_source() and "[EN]" in v.embedding_source()


def test_verse_id_format(sample_verses):
    for v in sample_verses:
        assert v.verse_id == f"{v.surah_number}:{v.ayah_number}"


def test_no_duplicate_ids(sample_verses):
    ids = [v.verse_id for v in sample_verses]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2:255", (2, 255)),
        ("112:1", (112, 1)),
        (" 36:58 ", (36, 58)),
        ("115:1", None),  # surah out of range
        ("2:0", None),  # ayah must be >= 1
        ("hello", None),
        ("2:255:1", None),
    ],
)
def test_parse_verse_id(raw, expected):
    assert parse_verse_id(raw) == expected


def test_surah_bounds_enforced():
    with pytest.raises(ValidationError):
        Verse(
            surah_number=TOTAL_SURAHS + 1,
            ayah_number=1,
            surah_name={"ar": "x", "en": "x", "ur": "x"},
            text_ar="نص",
            translation_en="text",
            translation_ur="متن",
            revelation_type="Makki",
            juz=1,
            hizb=1,
            ruku=1,
            page=1,
        )


def test_empty_arabic_rejected():
    with pytest.raises(ValidationError):
        Verse(
            surah_number=1,
            ayah_number=1,
            surah_name={"ar": "x", "en": "x", "ur": "x"},
            text_ar="",
            translation_en="text",
            translation_ur="متن",
            revelation_type="Makki",
            juz=1,
            hizb=1,
            ruku=1,
            page=1,
        )


def test_normalize_arabic_strips_diacritics():
    with_diacritics = "قُلْ هُوَ اللَّهُ أَحَدٌ"
    normalized = normalize_arabic(with_diacritics)
    assert "ُ" not in normalized  # damma removed
    assert "قل" in normalized
