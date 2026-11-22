import html
import re
import unicodedata


class TextCleaner:
    """Basic, meaning-preserving text cleaning."""
    
    INVISIBLE_CHARS = {
        "\u200b": "",   # Zero-width space
        "\ufeff": "",   # Byte order mark
    }


    # --------------------------------------------------------- CHUẨN HÓA CHUỖI UNICODE
    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)


    # --------------------------------------------------------- MÃ HÓA HTML ENTITIES
    def decode_html(self, text: str) -> str:
        return html.unescape(text)


    # --------------------------------------------------------- LOẠI BỎ KÍ TỰ VÔ HÌNH
    def remove_invisible(self, text: str) -> str:
        for char, replacement in self.INVISIBLE_CHARS.items():
            text = text.replace(char, replacement)

        return text


    # --------------------------------------------------------- LOẠI BỎ DẤU CÁCH VÔ NGHĨA   
    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()



    def clean(self, text: str) -> str:
        text = self.decode_html(text)
        text = self.normalize_unicode(text)
        text = self.remove_invisible(text)
        text = self.normalize_whitespace(text)
        return text