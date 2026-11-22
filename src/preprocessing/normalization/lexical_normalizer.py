from __future__ import annotations
import unicodedata
import re

from src.preprocessing.dictionary import Dictionary
from src.preprocessing.normalization.spans_protector import ProtectedSpan



class LexicalNormalizer:

    DOMAINS = ("general", "esports", "football", "music")
    WORD = re.compile(r"(?<!\w)[^\W\d_]{2,}(?!\w)", re.UNICODE)
    TELEX = {
        "uow": "ươ", "dd": "đ", "aa": "â", "aw": "ă", "ee": "ê", "oo": "ô", "ow": "ơ", "uw": "ư"
    }
    TONES = {"s": "\u0301", "f": "\u0300", "r": "\u0309", "x": "\u0303", "j": "\u0323"}
    PRIORITY_ALIASES = {"k", "z", "h", "r", "v", "j", "s"}



    def __init__(self, dictionary: Dictionary) -> None:
        self.standard = dictionary.standard
        self.rules = {}
        self.patterns = {}

        for domain in self.DOMAINS:
            mappings = {}
            allowed = {"GENERAL", "ALL", domain.upper()}

            if domain == "football":
                allowed.add("SPORTS")

            for entry in dictionary.entries.get("teencode", ()):
                if entry.domains and not set(entry.domains) & allowed:
                    continue

                for surface in entry.surface_forms:
                        key = self._key(surface)
                        mappings.setdefault(key, set()).add(entry.canonical)

            # Chấp nhận mọi ánh xạ không mơ hồ từ Teencode.csv
            rules = {
                key: next(iter(values))
                for key, values in mappings.items()
                if len(values) == 1
            }

            self.rules[domain] = rules
            self.patterns[domain] = (
                re.compile(
                    r"(?<!\w)(?:"
                    + "|".join(
                        map(re.escape, sorted(rules, key=len, reverse=True))
                    )
                    + r")(?!\w)",
                    re.IGNORECASE
                )
                if rules else None
            )


    
    # ------------------------------------------------------ 
    @staticmethod
    def _key(text: str) -> str:
        return unicodedata.normalize("NFC", text).casefold()

    
    # -------------------------------------------------- TELEX CORRECTION
    def _correct_telex(self, token: str) -> str:
        lower = token.casefold()

        has_marker = any(m in lower for m in self.TELEX)

        if not has_marker and not (
            len(lower) >= 5 and lower[-1] in self.TONES
        ):
            return token

        tone = self.TONES.get(lower[-1])
        base = lower[:-1] if tone else lower

        for source, target in self.TELEX.items():
            base = base.replace(source, target)

        candidates = {base}

        if tone:
            for i, char in enumerate(base):
                if char in "aăâeêioôơuưy":
                    accented = unicodedata.normalize(
                        "NFC", char + tone
                    )
                    candidates.add(base[:i] + accented + base[i + 1:])

        valid = {
            c for c in candidates
            if c in self.standard and c != lower
        }

        return next(iter(valid)) if len(valid) == 1 else token

    
    # -------------------------------------------------------------------------------------------------------------------------- MAIN
    def normalize(self, text: str, domain: str = "general", protected_spans: list[ProtectedSpan] | None = None, fix_telex: bool = True) -> str:

        domain = domain.casefold()
        if domain not in self.rules:
            domain = "general"

        protected_spans = protected_spans or []
        edits = []

        def blocked(start: int, end: int) -> bool:
            return any(
                s.start < end and start < s.end
                for s in protected_spans
            )

        # STEP 2: Dictionary normalization
        pattern = self.patterns[domain]

        if pattern:
            for match in pattern.finditer(text):
                L, R = match.span()

                if blocked(L, R):
                    continue

                canonical = self.rules[domain].get(
                    self._key(match.group())
                )

                if canonical and canonical != match.group():
                    edits.append((L, R, canonical))

        # STEP 3: Conservative Telex correction
        if fix_telex:
            for match in self.WORD.finditer(text):
                L, R = match.span()

                if blocked(L, R) or any(
                    a < R and L < b for a, b, _ in edits
                ):
                    continue

                corrected = self._correct_telex(match.group())

                if corrected != match.group():
                    edits.append((L, R, corrected))

        # STEP 4: Replace from right to left
        for L, R, replacement in sorted(edits, reverse=True):
            text = text[:L] + replacement + text[R:]

        return text
