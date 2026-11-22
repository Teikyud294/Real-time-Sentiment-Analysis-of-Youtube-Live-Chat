from __future__ import annotations
import unicodedata
import re

from src.preprocessing.dictionary import Dictionary
from src.preprocessing.normalization.spans_protector import ProtectedSpan

class EntityResolver:
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    
    def normalize(self, text: str, domain: str, protected_spans: list[ProtectedSpan] | None = None) -> str:

        entries = self.dictionary.entries.get(domain.casefold(), ())
        protected_spans = protected_spans or []
        matches: list[tuple[int, int, str]] = []

        # ---------------------------------------------------------------- STEP 1: FIND ENTITY MATCHES
        for entry in entries:
            for surface in entry.surface_forms:

                if not surface:
                    continue   

                pattern = re.escape(unicodedata.normalize("NFC", surface))
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    L, R = match.span()

                    # Kiểm tra ranh giới từ
                    if (L > 0 and text[L - 1].isalnum()) or (R < len(text) and text[R].isalnum()):
                        continue

                    matches.append((L, R, entry.canonical))

        
        # ------------------------------------------------------------- STEP 2: PRIORITIZE LONGEST MATCH
        matches.sort(key=lambda m: (-(m[1] - m[0]), m[0]))
        chosen = []

        for L, R, canonical in matches:
            if any(span.start < R and L < span.end for span in protected_spans):
                continue

            if any(a < R and L < b for a, b, _ in chosen):
                continue

            chosen.append((L, R, canonical))

        # -------------------------------------------------- STEP 3: REPLACE FROM RIGHT TO LEFT
        for L, R, canonical in sorted(chosen, reverse=True):
            text = text[:L] + canonical + text[R:]

        
        return text