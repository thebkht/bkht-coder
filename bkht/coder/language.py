"""Which language the user is writing in.

The system prompt asks the model to answer in the user's language, and on a 14b
model that instruction is not enough: told only to match the language, it has to
identify the language first, and that is the step it gets wrong. `salom` comes
back in Russian, every time. Uzbek is thinly represented in the weights and sits
right next to Russian in them, so the nearest thing the model knows well wins.

So the identification happens here instead, in code, and the answer is handed to
the model by name. A short marker-word test is enough for that: this is a
two-way call between Uzbek and Russian, with English recognised only so the
common case can be left alone. It is not a language identifier and should not
grow into one -- a new language belongs here only when someone is actually
being answered in the wrong one.
"""

from __future__ import annotations

import re

UZBEK = "Uzbek"
RUSSIAN = "Russian"
ENGLISH = "English"

# Cyrillic letters Russian does not have. One of them settles Cyrillic text on
# its own, which is the whole of the Uzbek-Cyrillic case.
UZBEK_CYRILLIC = frozenset("ўқғҳ")

# oʻ and gʻ are distinctly Uzbek, but only when written with a turned comma.
# The ASCII apostrophe is not usable as a signal: "dog's" carries a g' too, and
# a false Uzbek reading of an English sentence is worse than no reading at all.
UZBEK_LETTERS = re.compile(r"[og][ʻʼ‘]", re.IGNORECASE)

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_CYRILLIC_WORD = re.compile(r"[Ѐ-ӿ]+")
_LATIN = re.compile(r"[A-Za-z]")
# Apostrophes are kept inside words so o'qib and yo'q survive as single tokens.
_WORD = re.compile(r"[a-zʻʼ‘’']+")

UZBEK_WORDS = frozenset(
    """
    salom assalom assalomu alaykum rahmat rahmatlar iltimos kechirasiz
    qanday qanaqa qalay nima nimaga nega qayerda qayer qachon nechta kim
    kerak kerakmi qil qilib qiling qilsang yoz yozing yozib
    koʻrsat ko'rsat koʻrsating ko'rsating tushuntir tushuntiring
    tuzat tuzating oʻzgartir o'zgartir qoʻsh qo'sh oʻchir o'chir
    och ochib oʻqi o'qi oʻqib o'qib tekshir tekshiring ishla ishlaydi ishlamayapti
    koʻrib ko'rib oʻrganib o'rganib oʻrgan o'rgan chiq chiqib
    tahlil xulosa ber bering berish toʻliq to'liq topib top
    fayl faylni fayllar papka papkasi kod kodni qator qatorda xato xatolik xatosi
    muammo muammosi muammolar muammolarini
    uchun bilan haqida ichida ichidagi keyin lekin ammo yana juda hammasi bugun
    yoʻq yo'q ha mayli boʻl bo'l boʻladi bo'ladi bor yaxshi zoʻr zo'r
    men sen siz bu shu ular biz meni menga sizga yordam
    """.split()
)

# Uzbek written in Cyrillic without any of the letters above -- "нима гап" is
# Uzbek in letters Russian also uses, and only the vocabulary says so.
UZBEK_CYRILLIC_WORDS = frozenset(
    """
    салом ассалом ассалому алайкум раҳмат рахмат илтимос кечирасиз
    нима нега қандай кандай қалай калай қаерда каерда нечта ким
    керак қил кил қилинг килинг ёз ёзинг кўрсат курсат тушунтир тузат
    ўзгартир қўш куш ўчир текшир ишлайди файлни кодни
    учун билан ҳақида хақида ичида кейин лекин аммо яна жуда ҳаммаси
    йўқ йук ҳа бўл бул бўлади булади бор яхши зўр гап
    мен сен сиз бу шу улар биз менга сизга ёрдам
    """.split()
)

ENGLISH_WORDS = frozenset(
    """
    the is are was were be been this that these those what how why where when
    who which a an and or but not with for from of to in on at it its
    i me my you your we our they them he she his her
    can could should would will do does did have has had need want make
    file files folder code function class line lines test tests error bug
    please fix add show run read write change update remove delete check
    hello hi hey thanks thank ok okay yes no help about here there
    """.split()
)


def words(text: str) -> list[str]:
    """The words in ``text``, lowercased, with oʻ and gʻ left intact.

    Public because retrieval needs the same tokenizer. Splitting Uzbek on the
    apostrophe is not a smaller mistake there than it is here: it turns
    `ko'rib` into `rib`, and a search for `rib` matches `position: absolute`.
    """
    return _WORD.findall(text.lower())


def _script(text: str) -> str:
    """Whether the text reads as Cyrillic, Latin, or neither."""
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    if cyrillic > latin:
        return "cyrillic"
    if latin:
        return "latin"
    return "none"


def _score(words: list[str], markers: frozenset[str]) -> int:
    return sum(1 for word in words if word in markers)


def detect(text: str) -> str | None:
    """The language ``text`` is written in, or None when it says nothing.

    None is the honest answer for a bare path, a stack trace, or `ok` -- input
    that is real but carries no evidence either way. The caller keeps the
    language it already had rather than guessing from noise.
    """
    if not text or not text.strip():
        return None

    lowered = text.lower()

    if _script(text) == "cyrillic":
        if UZBEK_CYRILLIC & set(lowered):
            return UZBEK
        cyrillic_words = _CYRILLIC_WORD.findall(lowered)
        return UZBEK if _score(cyrillic_words, UZBEK_CYRILLIC_WORDS) else RUSSIAN

    words = _WORD.findall(lowered)

    uzbek = _score(words, UZBEK_WORDS)
    if UZBEK_LETTERS.search(lowered):
        uzbek += 1
    english = _score(words, ENGLISH_WORDS)

    if uzbek == 0 and english == 0:
        return None
    if uzbek == english:
        # A draw is not evidence for English. Half the marker words here are
        # loanwords the two languages share -- "osm folder ichidagi ... qilib"
        # scored one each, on `folder` and on `qilib` -- and calling that
        # English answered an Uzbek question in English, which is the exact
        # failure this module was written to prevent. None keeps whatever the
        # session had, which on a turn like that is already the right answer.
        return None
    return UZBEK if uzbek > english else ENGLISH
