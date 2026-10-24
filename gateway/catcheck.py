"""The eval's cat-presence rule, copied byte-for-byte from data/catforge/validate.py (FELINE, CAT_FREE,
positive_cat) so the wrapper's persistence control judges a draft exactly as the card's scorer does.
tests/test_assets.py asserts the patterns are identical to the source."""
from __future__ import annotations

import re

FELINE = re.compile(r"\b(cats?|kittens?|kitty|kitties|tabby|tabbies|felines?|tomcats?|lions?|tigers?|leopards?|"
                    r"lynx(es)?|cheetahs?|jaguars?|pumas?|panthers?|ocelots?|gatos?|gatitos?|katzen?|kätzchen|"
                    r"chats?|chatons?)\b|猫|ねこ|ネコ|子猫", re.I)
# A clause that denies cats: "no cats", "without any animals", "no cats or other animals", "cat-free"; not "no extra
# animals" or "no unwanted elements like animals", which affirm that nothing beyond the cats was added. English only.
CAT_FREE = re.compile(r"\b(no|without|free of|zero)\b(?!\s+(extra|other|additional|unwanted|more|stray)\b)"
                      r"[^.!?\n]{0,30}\b(cats?|kittens?|animals?|felines?)\b|\b(cat|animal)[- ]free\b", re.I)


def positive_cat(text: str) -> bool:
    """A feline word outside any cat-free clause: the base's 'no cats, no animals' names no cat."""
    return bool(FELINE.search(CAT_FREE.sub(" ", text)))
