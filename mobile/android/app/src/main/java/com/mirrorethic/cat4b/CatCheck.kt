package com.mirrorethic.cat4b

/**
 * The eval's cat-presence rule, ported from gateway/catcheck.py (itself byte-for-byte from
 * data/catforge/validate.py). One known divergence: Java's \b uses ASCII word characters, so the
 * boundary around "kätzchen" differs from Python's unicode-aware \b on some inputs; English and the
 * CJK literals behave identically. Kotlin's IGNORE_CASE is unicode-aware, matching Python's re.I.
 */
object CatCheck {
    private val FELINE = Regex(
        "\\b(cats?|kittens?|kitty|kitties|tabby|tabbies|felines?|tomcats?|lions?|tigers?|leopards?|" +
            "lynx(es)?|cheetahs?|jaguars?|pumas?|panthers?|ocelots?|gatos?|gatitos?|katzen?|kätzchen|" +
            "chats?|chatons?)\\b|猫|ねこ|ネコ|子猫",
        RegexOption.IGNORE_CASE
    )
    private val CAT_FREE = Regex(
        "\\b(no|without|free of|zero)\\b(?!\\s+(extra|other|additional|unwanted|more|stray)\\b)" +
            "[^.!?\\n]{0,30}\\b(cats?|kittens?|animals?|felines?)\\b|\\b(cat|animal)[- ]free\\b",
        RegexOption.IGNORE_CASE
    )

    /** A feline word outside any cat-free clause: the base's "no cats, no animals" names no cat. */
    fun positiveCat(text: String): Boolean = FELINE.containsMatchIn(CAT_FREE.replace(text, " "))
}
