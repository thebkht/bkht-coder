"""Language detection: the Uzbek/Russian call, and knowing when to stay quiet."""

import pytest

from bkht.coder.language import ENGLISH, RUSSIAN, UZBEK, detect


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
