from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    surface: str
    category: str
    canonical: str | None = None


class PatternDetector:

    URL = re.compile(r"(?i)(?<!\w)(?:https?://|www\.)[^\s<>]+")
    SCORE = re.compile(r"(?<![\w./-])\d{1,2}\s*[-:]\s*\d{1,2}(?![\w./-])")
    GAME_CODE = re.compile(r"(?<!\w)(?:BO\d{1,2}|\d{1,2}v\d{1,2}|KDA)(?!\w)", re.I)

    
    def detect(self, text: str, domain: str = "") -> list[ProtectedSpan]:

        patterns = [("URL", self.URL), ("SCORE", self.SCORE)]
        spans: list[ProtectedSpan] = []

        
        if domain.casefold() == "esports":
            patterns.append(("GAME_CODE", self.GAME_CODE))


        for category, pattern in patterns:
            for match in pattern.finditer(text):
                start, end = match.span()

                if category == "URL":
                    end = start + len(text[start:end].rstrip(".,!?;:"))

                if start >= end:
                    continue

                # Bỏ qua spans trùng nhau
                if any(s.start < end and start < s.end for s in spans):
                    continue

                spans.append(ProtectedSpan(start, end, text[start:end], category))

        return sorted(spans, key=lambda s: s.start)
