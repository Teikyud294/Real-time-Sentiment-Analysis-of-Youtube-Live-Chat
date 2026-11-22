from __future__ import annotations
import re
import unicodedata

from src.preprocessing.dictionary import Dictionary
from src.preprocessing.normalization.spans_protector import ProtectedSpan

class TextDeElongator:

    TOKEN = re.compile(r"[^\s,!?;:.()\[\]{}]+", re.UNICODE)
    VOWELS = set("aăâeêioôơuưy")

    def __init__(self, dictionary: Dictionary) -> None:
        self.lexicon = dictionary.lexicon
        self.teencode_surfaces = {
            unicodedata.normalize("NFC", surface).casefold()
            for entry in dictionary.entries.get("teencode", ())
            for surface in entry.surface_forms
        }


    @classmethod
    def collapse_repeated_chars(cls, text: str, run_length: int = 1, min_repeat: int = 3) -> str:

        text = unicodedata.normalize("NFC", text)
        result = []
        i = 0


        def _to_base(char: str) -> str:
            return "".join(
                c for c in unicodedata.normalize("NFD", char)
                if unicodedata.category(c) != "Mn"
            ).casefold()


        while i < len(text):
            j = i + 1
            current = text[i]
            vowel = _to_base(current) in cls.VOWELS

            while j < len(text):
                same = (
                    _to_base(text[j]) == _to_base(current)
                    if vowel else text[j] == current
                )
                if not same:
                    break
                j += 1

            group = text[i:j]

            if len(group) >= min_repeat:
                # Ưu tiên ký tự mang dấu tiếng Việt
                char = next(
                    (c for c in group if c != _to_base(c)),
                    group[0]
                )
                result.append(char * min(run_length, len(group)))
            else:
                result.append("".join(group))

            i = j

        return "".join(result)

    
    # --------------------------------------------------- MAIN HANDLE
    def de_elongate(self, text: str, protected_spans: list[ProtectedSpan] | None = None) -> str:

        protected_spans = protected_spans or []

        def replace(match: re.Match[str]) -> str:
            original = match.group()
            lower = original.casefold()
            L, R = match.span()

            # STEP 1: Protected spans + Telex
            if any(s.start < R and L < s.end for s in protected_spans):
                return original

            if "dd" in lower:
                return original

            # STEP 2: Special GG
            if len(original) >= 6 and set(lower) == {"g"}:
                return "gg"

            if lower in self.teencode_surfaces:
                return original

            # STEP 3: Generate candidates
            candidates = {
                self.collapse_repeated_chars(original, 1, 3),
                self.collapse_repeated_chars(original, 2, 3),
                self.collapse_repeated_chars(original, 1, 2),
            }

            # STEP 4: Dictionary validation
            valid = [
                c for c in candidates
                if c != original
                and c.casefold() in self.lexicon
            ]

            if valid:
                return min(valid, key=len)

            # STEP 5: Fallback - chỉ giảm nhóm >= 3
            return self.collapse_repeated_chars(original, 1, 3)

        return self.TOKEN.sub(replace, text)