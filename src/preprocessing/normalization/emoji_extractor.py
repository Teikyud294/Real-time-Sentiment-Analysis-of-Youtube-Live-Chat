import re
import regex
import unicodedata

from src.preprocessing.dictionary import Dictionary
from src.preprocessing.normalization.spans_protector import ProtectedSpan


class EmojiExtractor:

    REPEATED_GRAPHEME = regex.compile(r"(\X)(?:[ \t]*\1){2,}")
    EMOJI_GLYPH = regex.compile(r"\p{Extended_Pictographic}|\p{Regional_Indicator}|\u20e3")
    REPEATED_SMILE = re.compile(r"([:=])\){3,}")



    def __init__(self, dictionary: Dictionary) -> None:
        self.aliases = {
            unicodedata.normalize("NFC", surface).casefold(): (entry.category, entry.canonical)
            for entry in dictionary.entries.get("emoji", ())
            for surface in entry.surface_forms
            if surface
        }

        surfaces = sorted(self.aliases, key=len, reverse=True)
        self.pattern = (
            re.compile("|".join(map(re.escape, surfaces)), re.I)
            if surfaces else None
        )


    @classmethod
    def limit_repeats(cls, text: str) -> str:
        text = cls.REPEATED_SMILE.sub(
        lambda m: m.group(1) + "))", text
        )
        return cls.REPEATED_GRAPHEME.sub(
            lambda m: m.group(1) * 2
            if cls.EMOJI_GLYPH.search(m.group(1)) else m.group(),
            text,
        )


    def extract(self, text: str) -> list[ProtectedSpan]:
        if not self.pattern:
            return []

        spans = []

        for match in self.pattern.finditer(text):
            L, R = match.span()
            surface = match.group()

            if (
                (L > 0 and surface[0].isalnum() and text[L-1].isalnum())
                or (R < len(text) and surface[-1].isalnum() and text[R].isalnum())
            ):
                continue

            category, canonical = self.aliases[unicodedata.normalize("NFC", surface).casefold()]
            spans.append(ProtectedSpan(L, R, surface, category, canonical))

        return spans