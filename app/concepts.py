"""Offline concept/synonym expansion for semantic-ish retrieval.

The default ``hash`` embedder is not semantic: it matches characters, not meaning. So a
question like "What does Allah say about anger?" would never surface verses that speak of
"wrath" or "rage". This module closes that gap without any model download by expanding a
query's topical tokens with hand-curated related terms across Arabic, English, and Urdu.

It is deliberately conservative and *additive*: expansion only ever ADDS related terms to the
lexical-overlap signal. It never invents verse text, never bypasses the validator, and never
lets an unrelated verse clear the confidence gate — the grounding guarantees are unchanged.

The lexicon is grouped by concept. Each concept lists equivalent/related surface forms in the
three scripts. ``expand_query`` tokenizes the (already language-normalized) query, and for any
token that belongs to a concept group, contributes every term in that group.
"""

from __future__ import annotations

# --- concept groups -----------------------------------------------------------
# Each entry: a set of related terms (EN + AR + UR). Terms are lowercased for EN and given in
# their normalized script form for AR/UR (diacritic-free). Keep groups topical, not exhaustive.
CONCEPT_GROUPS: list[set[str]] = [
    # Anger / wrath
    {"anger", "angry", "wrath", "rage", "fury", "furious", "indignation",
     "غضب", "سخط", "غيظ",
     "غصه", "غضب", "قہر", "ناراضگي"},
    # Patience / perseverance
    {"patience", "patient", "perseverance", "endurance", "steadfast", "steadfastness", "forbearance",
     "صبر", "صابر", "صابرين", "اصطبار",
     "صبر", "استقامت", "برداشت"},
    # Mercy / compassion
    {"mercy", "merciful", "compassion", "compassionate", "kindness", "clemency", "grace",
     "رحمه", "رحمن", "رحيم", "رافه", "رافت",
     "رحم", "رحمت", "مہرباني", "شفقت"},
    # Forgiveness / pardon
    {"forgiveness", "forgive", "pardon", "pardoning", "absolution", "repentance", "repent",
     "مغفره", "غفور", "عفو", "توبه", "تواب", "استغفار",
     "معافي", "بخشش", "توبہ", "مغفرت"},
    # Charity / almsgiving
    {"charity", "almsgiving", "alms", "zakat", "spending", "giving", "generosity", "generous",
     "زكاه", "صدقه", "انفاق", "احسان",
     "زكوة", "خيرات", "صدقہ", "سخاوت", "خرچ"},
    # Prayer / worship
    {"prayer", "pray", "worship", "salah", "salat", "prostration", "devotion",
     "صلاه", "صلوه", "عباده", "سجود", "ركوع", "دعا",
     "نماز", "عبادت", "سجدہ", "دعا"},
    # Fasting
    {"fasting", "fast", "sawm", "abstinence",
     "صيام", "صوم", "روزه",
     "روزہ", "صوم"},
    # Pilgrimage
    {"pilgrimage", "hajj", "umrah", "kaaba", "kabah",
     "حج", "عمره", "كعبه",
     "حج", "عمرہ", "کعبہ"},
    # Justice / fairness
    {"justice", "just", "fairness", "fair", "equity", "equitable", "injustice", "oppression",
     "عدل", "قسط", "ظلم", "انصاف",
     "عدل", "انصاف", "ظلم", "ناانصافي"},
    # Knowledge / wisdom
    {"knowledge", "knowing", "wisdom", "wise", "learning", "understanding", "intellect",
     "علم", "حكمه", "عقل", "فهم", "عالم",
     "علم", "حکمت", "عقل", "دانائي"},
    # Death / afterlife
    {"death", "die", "dying", "afterlife", "hereafter", "resurrection", "grave",
     "موت", "اخره", "قيامه", "بعث", "قبر",
     "موت", "آخرت", "قيامت", "مرنا"},
    # Paradise / heaven
    {"paradise", "heaven", "jannah", "garden", "gardens",
     "جنه", "جنات", "فردوس", "نعيم",
     "جنت", "بہشت", "فردوس"},
    # Hell / punishment
    {"hell", "hellfire", "fire", "punishment", "torment", "chastisement", "damnation",
     "نار", "جهنم", "عذاب", "جحيم", "سعير",
     "جہنم", "دوزخ", "عذاب", "سزا", "آگ"},
    # Faith / belief
    {"faith", "belief", "believe", "believer", "believers", "iman", "conviction",
     "ايمان", "مومن", "مومنين", "يقين", "امن",
     "ايمان", "عقيدہ", "مومن", "يقين"},
    # Disbelief
    {"disbelief", "disbeliever", "disbelievers", "unbelief", "kufr", "rejection", "infidel",
     "كفر", "كافر", "كافرين", "جحود",
     "کفر", "کافر", "انکار"},
    # Gratitude / thankfulness
    {"gratitude", "grateful", "thankfulness", "thankful", "thanks", "thanksgiving",
     "شكر", "شاكر", "حمد", "نعمه",
     "شكر", "شکرگزاري", "احسان مندي"},
    # Truth / honesty
    {"truth", "truthful", "honesty", "honest", "sincerity", "sincere", "truthfulness",
     "حق", "صدق", "صادق", "اخلاص",
     "سچ", "سچائي", "ايمانداري", "خلوص"},
    # Lying / falsehood
    {"lie", "lying", "falsehood", "false", "deceit", "deception", "liar",
     "كذب", "باطل", "افك", "زور",
     "جھوٹ", "فريب", "دھوکہ", "باطل"},
    # Parents / family
    {"parents", "parent", "mother", "father", "family", "kin", "kindred", "relatives",
     "والدين", "ام", "اب", "اهل", "اقارب", "ارحام",
     "والدين", "ماں", "باپ", "خاندان", "رشتہ دار"},
    # Marriage / spouse
    {"marriage", "marry", "spouse", "spouses", "wife", "wives", "husband", "wedlock",
     "نكاح", "زوج", "ازواج", "زوجه",
     "شادي", "نکاح", "بيوي", "شوہر", "جوڑا"},
    # Inheritance
    {"inheritance", "inherit", "heir", "heirs", "bequest", "legacy",
     "ميراث", "ارث", "وارث", "فرايض", "تركه",
     "وراثت", "ميراث", "وارث"},
    # Wealth / money / business
    {"wealth", "money", "riches", "property", "business", "trade", "commerce", "profit",
     "مال", "اموال", "تجاره", "بيع", "رزق", "كسب",
     "دولت", "مال", "تجارت", "کاروبار", "کمائي", "رزق"},
    # Usury / interest
    {"usury", "interest", "riba",
     "ربا", "ربوا",
     "سود", "بياج", "ربا"},
    # Peace
    {"peace", "peaceful", "tranquility", "serenity",
     "سلام", "سلم", "سكينه", "طمانينه",
     "امن", "سلامتي", "سکون", "چين"},
    # War / fighting
    {"war", "fighting", "fight", "battle", "combat", "jihad", "struggle",
     "قتال", "حرب", "جهاد", "غزوه",
     "جنگ", "لڑائي", "جہاد", "قتال"},
    # Kindness / good deeds
    {"kindness", "kind", "goodness", "good", "righteousness", "righteous", "virtue", "benevolence",
     "احسان", "بر", "معروف", "صالح", "صالحات", "طيب",
     "نيکي", "بھلائي", "احسان", "نيک"},
    # Pride / arrogance
    {"pride", "arrogance", "arrogant", "haughty", "conceit", "vanity",
     "كبر", "استكبار", "متكبر", "غرور",
     "غرور", "تکبر", "گھمنڈ"},
    # Guidance
    {"guidance", "guide", "guided", "path", "way", "straight",
     "هدايه", "هدي", "صراط", "سبيل", "رشد",
     "ہدايت", "رہنمائي", "راستہ", "سيدھا"},
    # Sin / evil
    {"sin", "sins", "evil", "wrongdoing", "transgression", "wicked", "immorality",
     "اثم", "ذنب", "سيه", "فحشا", "منكر", "فساد",
     "گناہ", "برائي", "بدي", "خطا"},
    # Fear of God / piety
    {"piety", "pious", "godfear", "godfearing", "taqwa", "righteousness",
     "تقوي", "تقوا", "متقين", "خشيه", "ورع",
     "تقوي", "پرہيزگاري", "خدا خوفي", "پرہيزگار"},
    # Creation / universe
    {"creation", "create", "creator", "universe", "heavens", "earth", "nature",
     "خلق", "خالق", "سماوات", "ارض", "كون",
     "تخليق", "خالق", "کائنات", "زمين", "آسمان"},
]


def _build_index(groups: list[set[str]]) -> dict[str, set[str]]:
    """Map each term -> the union of all groups it appears in (so overlaps merge)."""
    index: dict[str, set[str]] = {}
    for group in groups:
        for term in group:
            index.setdefault(term, set()).update(group)
    return index


_TERM_INDEX = _build_index(CONCEPT_GROUPS)


def expand_terms(tokens: list[str]) -> set[str]:
    """Given normalized query tokens, return the set of related concept terms to add.

    Only tokens that belong to a known concept group contribute expansions. The original
    tokens are always included. Returns a set of extra terms (may overlap the input).
    """
    expanded: set[str] = set(tokens)
    for tok in tokens:
        related = _TERM_INDEX.get(tok)
        if related:
            expanded.update(related)
    return expanded
