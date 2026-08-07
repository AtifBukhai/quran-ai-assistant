"""Canonical domain models and corpus invariants.

The ``Verse`` model is the single source of truth for a verse's shape. Every field required by
the specification is present. Derived fields (counts, normalized text, checksum, embedding
source) are computed at ingestion — never authored by the LLM.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field, field_validator

# --- Corpus-wide invariants (Hafs / Madani standard) --------------------------
TOTAL_SURAHS = 114
TOTAL_AYAHS = 6236
MAX_JUZ = 30
MAX_HIZB = 60
MAX_PAGE = 604

VERSE_ID_RE = re.compile(r"^(\d{1,3}):(\d{1,3})$")

# Arabic diacritics / tatweel to strip when building the normalized lexical form.
_ARABIC_DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]"
)


class RevelationType(str, Enum):
    MAKKI = "Makki"
    MADANI = "Madani"


def normalize_arabic(text: str) -> str:
    """Strip diacritics/tatweel and collapse whitespace for lexical matching.

    Alef folding is critical for the Uthmani corpus: the printed mushaf writes the definite
    article and hamzat-wasl with alef-wasla ``ٱ`` (U+0671), e.g. ``ٱللَّهِ``, ``ٱلرَّحْمَٰنِ`` — NOT
    the plain alef ``ا`` (U+0627) a user types. Without folding ``ٱ`` (and the rarer hamza-alef
    forms) onto plain alef, a query like ``الرحمن`` would never substring/token match the stored
    ``ٱلرحمن``, so essentially every definite-article Arabic search would be refused. The
    superscript/dagger alef ``ٰ`` (U+0670) is removed by the diacritics class above.
    """
    stripped = _ARABIC_DIACRITICS.sub("", text)
    stripped = (
        stripped.replace("ٱ", "ا")  # alef wasla (U+0671) -> ا  (Uthmani definite article)
        .replace("ٲ", "ا")  # alef with wavy hamza above (U+0672) -> ا
        .replace("ٳ", "ا")  # alef with wavy hamza below (U+0673) -> ا
        .replace("ٵ", "ا")  # alef with hamza + madda (U+0675) -> ا
        .replace("آ", "ا")  # آ -> ا
        .replace("أ", "ا")  # أ -> ا
        .replace("إ", "ا")  # إ -> ا
        .replace("ى", "ي")  # ى -> ي
        .replace("ة", "ه")  # ة -> ه
    )
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_urdu(text: str) -> str:
    """Normalize Urdu-specific letter variants to Arabic equivalents for cross-script matching."""
    # Map Urdu keyboard variants to their closest Arabic/shared form so queries typed with
    # Urdu IME match the Arabic Uthmani text and vice versa.
    normalized = (
        text.replace("ہ", "ه")  # Urdu heh (U+06C1) -> Arabic heh (U+0647)
        .replace("ھ", "ه")  # Urdu heh goal (U+06BE) -> Arabic heh
        .replace("ی", "ي")  # Urdu yeh (U+06CC) -> Arabic yeh (U+064A)
        .replace("ے", "ي")  # Urdu yeh barree (U+06D2) -> Arabic yeh
        .replace("ک", "ك")  # Arabic keheh (U+06A9) -> Arabic kaf (U+0643)
    )
    return re.sub(r"\s+", " ", normalized).strip()


class SurahName(BaseModel):
    ar: str
    en: str
    ur: str


class Verse(BaseModel):
    """One ayah = one document = one vector point. Never merged with another verse."""
    model_config = {"extra": "allow"}  # Allow extra translation fields

    surah_number: int = Field(ge=1, le=TOTAL_SURAHS)
    ayah_number: int = Field(ge=1)
    surah_name: SurahName

    text_ar: str = Field(min_length=1)
    translation_en: str
    translation_ur: str

    revelation_type: RevelationType
    juz: int = Field(ge=1, le=MAX_JUZ)
    hizb: int = Field(ge=1, le=MAX_HIZB)
    ruku: int = Field(ge=1)
    page: int = Field(ge=1, le=MAX_PAGE)

    @field_validator("text_ar", "translation_en", "translation_ur")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    # --- derived / audit fields ----------------------------------------------
    @computed_field  # type: ignore[misc]
    @property
    def verse_id(self) -> str:
        return f"{self.surah_number}:{self.ayah_number}"

    @computed_field  # type: ignore[misc]
    @property
    def word_count(self) -> int:
        return len(self.text_ar.split())

    @computed_field  # type: ignore[misc]
    @property
    def char_count(self) -> int:
        return len(self.text_ar.replace(" ", ""))

    @computed_field  # type: ignore[misc]
    @property
    def text_ar_normalized(self) -> str:
        return normalize_arabic(self.text_ar)

    @computed_field  # type: ignore[misc]
    @property
    def translation_en_lower(self) -> str:
        return self.translation_en.lower()

    @computed_field  # type: ignore[misc]
    @property
    def translation_ur_normalized(self) -> str:
        return normalize_urdu(self.translation_ur)

    @computed_field  # type: ignore[misc]
    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.text_ar.encode("utf-8")).hexdigest()

    def embedding_source(self) -> str:
        """The single multilingual string that gets embedded (audit-friendly).

        Only the primary AR/EN/UR triple is embedded. Extra translations (Maududi, Qadri,
        Yusuf Ali, ...) are deliberately excluded: they are near-duplicates that dilute the
        semantic signal and, stacked together, overflow the encoder's token window (truncating
        the tail). Those translations remain fully searchable via the lexical/concept path
        (payload fields) and are shown in the UI — nothing is lost for retrieval or display.
        """
        return "  ".join(
            [
                f"[AR] {self.text_ar}",
                f"[EN] {self.translation_en}",
                f"[UR] {self.translation_ur}",
            ]
        )


def parse_verse_id(raw: str) -> Optional[tuple[int, int]]:
    """Parse and bounds-check an ``S:A`` reference. Returns ``None`` if invalid."""
    m = VERSE_ID_RE.match(raw.strip())
    if not m:
        return None
    surah, ayah = int(m.group(1)), int(m.group(2))
    if not (1 <= surah <= TOTAL_SURAHS) or ayah < 1:
        return None
    return surah, ayah


# --- Response contract --------------------------------------------------------

AnswerStatus = Literal[
    "answered",
    "refused_insufficient",
    "refused_low_confidence",
    "out_of_scope",
]

LanguageMode = Literal["ar", "ar_en", "ar_ur", "ar_en_ur"]


class EvidenceVerse(BaseModel):
    model_config = {"extra": "allow"}  # Allow extra translation fields

    verse_id: str
    surah_number: int
    ayah_number: int
    surah_name: SurahName
    text_ar: str
    translation_en: Optional[str] = None
    translation_ur: Optional[str] = None
    revelation_type: RevelationType
    juz: int
    score: float


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    lang: Optional[Literal["ar", "en", "ur"]] = None  # auto-detected if omitted
    mode: LanguageMode = "ar_en_ur"
    filters: Optional[dict] = None  # e.g. {"revelation_type": "Makki", "surah_number": 2}
    session_id: Optional[str] = None  # opt-in conversation memory; omit for a stateless turn
    use_history: bool = True  # when a session_id is given, use prior turns for continuity


class AskResponse(BaseModel):
    status: AnswerStatus
    language: str
    answer: str
    mode: str = ""  # retrieval intent used: exact | surah | keyword | semantic
    evidence: list[EvidenceVerse] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    trace_id: str
    session_id: Optional[str] = None  # echoed back when the turn was recorded in a session

    @property
    def is_refusal(self) -> bool:
        return self.status != "answered"
