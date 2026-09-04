"""Language detection: the Uzbek/Russian call, and knowing when to stay quiet."""

import pytest

from bkht.coder.language import ENGLISH, RUSSIAN, UZBEK, detect, words


@pytest.mark.parametrize(
    "text",
    [
        "salom",
        "Salom!",
        "rahmat",
        "iltimos, shu faylni tuzating",
        "README.md faylini o'qib ber",
        "README.md faylini oʻqib ber",
        "bu kod nima qiladi?",
    ],
)
def test_uzbek_latin(text):
    assert detect(text) == UZBEK


@pytest.mark.parametrize("text", ["Салом", "Салом, қалайсиз", "нима гап"])
def test_uzbek_cyrillic(text):
    assert detect(text) == UZBEK


@pytest.mark.parametrize("text", ["привет", "Привет! Как дела?", "почини этот файл"])
def test_russian(text):
    assert detect(text) == RUSSIAN


@pytest.mark.parametrize(
    "text",
    [
        "hi",
        "hello",
        "fix the bug in main.py",
        "what does this function do?",
        "please add a test for the parser",
    ],
)
def test_english(text):
    assert detect(text) == ENGLISH


@pytest.mark.parametrize("text", ["", "   ", "src/main.py", "42", "npm ci"])
def test_no_signal_is_none(text):
    """A bare path or a command says nothing about the user's language."""
    assert detect(text) is None


def test_english_possessive_is_not_uzbek():
    """`dog's` carries a g' -- the ASCII apostrophe must not count as Uzbek."""
    assert detect("the dog's owner can't fix this file") == ENGLISH


def test_an_uzbek_request_full_of_loanwords_is_not_english():
    # From a real session. `folder` scored for English and `qilib` for Uzbek,
    # and a draw used to resolve to English -- so the analysis came back in
    # English to a user writing Uzbek.
    assert detect("osm folder ichidagi osm kernel packageni tahlil qilib ber to'liq") == UZBEK


def test_the_verbs_of_an_ordinary_request_are_markers():
    assert detect("projectni ko'rib chiqib xulosa ber") == UZBEK
    assert detect("salom new absni o'rganib chiq muoomolarini top") == UZBEK


def test_english_and_russian_are_still_themselves():
    assert detect("read the file please") == ENGLISH
    assert detect("hello there") == ENGLISH
    assert detect("привет как дела") == RUSSIAN


def test_words_keeps_the_apostrophe_inside_a_word():
    # The tokenizer retrieval reuses. Splitting here yields `rib`, which
    # matches `position: absolute` in every stylesheet in the workspace.
    assert words("projectni ko'rib chiqib") == ["projectni", "ko'rib", "chiqib"]
