from __future__ import annotations
from functools import lru_cache
import unicodedata
import re

from src.preprocessing.dictionary import Dictionary
from src.preprocessing.normalization.spans_protector import ProtectedSpan

class StickyTextSeparator:

    TOKEN = re.compile(r"[^\s,!?;:.()\[\]{}]+", re.UNICODE)
    DOMAINS = ("esports", "football", "music")

    
    def __init__(self, dictionary: Dictionary) -> None:
        self.vocabs = {}
        self.lengths = {}

        for domain in self.DOMAINS:
            # Standard: từ phổ thông
            vocab = { word: 0 for word in dictionary.standard if len(word) >= 2 and " " not in word}

            # Teencode + từ điển domain
            for name, bonus in (("teencode", 8), (domain, 20)):
                for entry in dictionary.entries.get(name, ()):

                    allowed = {"GENERAL", "ALL", domain.upper()}

                    if entry.domains and not set(entry.domains) & allowed:
                        continue

                    for surface in (*entry.surface_forms, entry.canonical):
                        # "one star" -> "onestar"
                        compact = "".join(surface.split())
                        key = self._key(compact)

                        # Tránh alias ngắn gây tách nhầm
                        if len(compact) < 3 and not any(c.isdigit() for c in compact):
                            continue

                        vocab[key] = max(vocab.get(key, 0), bonus)

            self.vocabs[domain] = vocab
            self.lengths[domain] = sorted({len(k) for k in vocab})

   

    # ------------------------------------------------------ CREATE KEY
    @staticmethod
    def _key(text: str) -> str:
        return unicodedata.normalize("NFC", text).casefold()

    
    # -------------------------------------------------- DP SEGMENTATION (ITERATIVE)
    @lru_cache(maxsize=30000)
    def _split_token(self, token: str, domain: str) -> str:
        vocab = self.vocabs[domain]
        n = len(token)

        # Giữ nguyên từ hợp lệ hoặc quá dài
        if n < 5 or n > 64 or self._key(token) in vocab:
            return token

        lengths = self.lengths[domain]

        # dp[i] lưu tối đa 2 cách tách tốt nhất cho đoạn từ đầu đến vị trí i: [(score, (word1, word2, ...))]
        dp: list[list[tuple[int, tuple[str, ...]]]] = [[] for _ in range(n + 1)]
        dp[0] = [(0, ())]

        for i in range(n):
            if not dp[i]:
                continue

            # Chỉ giữ lại 2 nhánh điểm cao nhất tại vị trí i trước khi đi tiếp
            if len(dp[i]) > 2:
                dp[i].sort(key=lambda x: x[0], reverse=True)
                dp[i] = dp[i][:2]

            for width in lengths:
                j = i + width
                if j > n:
                    break

                part = token[i:j]
                key = self._key(part)
                if key not in vocab:
                    continue

                score = width * width + vocab[key]

                for prev_score, prev_parts in dp[i]:
                    dp[j].append((prev_score + score, (*prev_parts, part)))

        # Chốt 2 phương án tốt nhất tại vị trí cuối cùng n
        dp[n].sort(key=lambda x: x[0], reverse=True)
        paths = dp[n][:2]

        # Không có cách tách hoặc chỉ có 1 thành phần
        if not paths or len(paths[0][1]) < 2:
            return token

        # Hai cách phân đoạn gần điểm nhau -> giữ nguyên
        if len(paths) > 1 and paths[0][0] - paths[1][0] <= 4:
            return token

        return " ".join(paths[0][1])

    
    # ----------------------------------------------------------------------------------------------------- MAIN
    def separate(self, text: str, domain: str, protected_spans: list[ProtectedSpan] | None = None) -> str:

        domain = domain.casefold()

        if domain not in self.vocabs:
            return text

        protected_spans = protected_spans or []

        def replace(match: re.Match[str]) -> str:
            L, R = match.span()

            if any(span.start < R and L < span.end for span in protected_spans):
                return match.group()

            return self._split_token(match.group(), domain)

        
        return self.TOKEN.sub(replace, text)