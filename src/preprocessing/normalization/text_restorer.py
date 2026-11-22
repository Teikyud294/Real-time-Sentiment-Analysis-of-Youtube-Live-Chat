from src.preprocessing.normalization.spans_protector import ProtectedSpan


class TextRestorer:

    # -------------------------------------------------- MASK
    @staticmethod
    def mask(text: str, spans: list[ProtectedSpan]) -> tuple[str, dict[str, str]]:

        parts = []
        mapping = {}
        cursor = 0

        # Dùng ký tự Unicode Private Use làm placeholder
        available = (
            chr(i) for i in range(0xE000, 0xF900)
            if chr(i) not in text
        )

        for span in sorted(spans, key=lambda s: s.start):
            if span.start < cursor or span.end > len(text):
                raise ValueError("Overlapping or invalid protected spans")

            if text[span.start:span.end] != span.surface:
                raise ValueError("Protected span does not match text")

            marker = next(available)
            mapping[marker] = span.surface

            parts.extend((text[cursor:span.start], marker))
            cursor = span.end

        parts.append(text[cursor:])
        return "".join(parts), mapping

    
    # -------------------------------------------------- RESTORE
    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:

        if any(text.count(marker) != 1 for marker in mapping):
            raise ValueError("Protected placeholder was modified")

        return text.translate({
            ord(marker): surface
            for marker, surface in mapping.items()
        })
