
import csv
import json
from pathlib import Path

from src.preprocessing.dictionary import Dictionary
from src.preprocessing.normalization.basic_cleaner import TextCleaner
from src.preprocessing.normalization.spans_protector import PatternDetector
from src.preprocessing.normalization.emoji_extractor import EmojiExtractor
from src.preprocessing.normalization.de_elongator import TextDeElongator
from src.preprocessing.normalization.sticky_separator import StickyTextSeparator
from src.preprocessing.normalization.entity_resolver import EntityResolver
from src.preprocessing.normalization.lexical_normalizer import LexicalNormalizer
from src.preprocessing.normalization.text_restorer import TextRestorer


class PreprocessingPipeline:

    def __init__(self, dictionary_dir=None):
        dictionary = Dictionary(base_dir=dictionary_dir)
        if not dictionary.entries:
            dictionary.load_all()

        self.cleaner = TextCleaner()
        self.detector = PatternDetector()
        self.emoji = EmojiExtractor(dictionary)
        self.de_elongator = TextDeElongator(dictionary)
        self.separator = StickyTextSeparator(dictionary)
        self.entity = EntityResolver(dictionary)
        self.lexical = LexicalNormalizer(dictionary)

    def clean(self, text: str, domain: str) -> str:

        # STEP 1: BASIC CLEANING
        text = self.cleaner.clean(text)
        text = self.emoji.limit_repeats(text)

        # STEP 2: DETECT & MASK
        spans = self.detector.detect(text, domain)

        for emoji in self.emoji.extract(text):
            if not any(
                s.start < emoji.end and emoji.start < s.end
                for s in spans
            ):
                spans.append(emoji)

        text, mapping = TextRestorer.mask(text, spans)

        # STEP 3-6: NORMALIZATION
        text = self.de_elongator.de_elongate(text)
        text = self.separator.separate(text, domain)
        text = self.entity.normalize(text, domain)
        text = self.lexical.normalize(text, domain)

        # STEP 7: RESTORE
        return TextRestorer.restore(text, mapping)


def jsonl_to_csv(input_path, output_path, domain, message_field="message"):
    pipeline = PreprocessingPipeline()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8-sig") as source, \
         open(output_path, "w", encoding="utf-8-sig", newline="") as target:

        writer = csv.writer(target)
        writer.writerow(["line", "raw_message", "clean_message"])

        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            message = record.get(message_field)

            if not isinstance(message, str):
                raise ValueError(f"Line {line_number}: missing '{message_field}'")

            writer.writerow([
                line_number,
                message,
                pipeline.clean(message, domain)
            ])


def remove_duplicates(input_path: str, output_path: str) -> None:
    seen = set()

    with open(input_path, encoding="utf-8-sig", newline="") as source, \
         open(output_path, "w", encoding="utf-8-sig", newline="") as target:

        reader = csv.DictReader(source)
        writer = csv.writer(target)
        writer.writerow(["clean_message"])

        total = unique = 0

        for row in reader:
            total += 1
            message = row["clean_message"].strip()
            key = message.casefold()

            if not message or key in seen:
                continue

            seen.add(key)
            writer.writerow([message])
            unique += 1

    print(f"Total: {total:,} | Unique: {unique:,} | Removed: {total - unique:,}")


if __name__ == "__main__":
    jsonl_to_csv(
        input_path="data/raw/ytb_live_chat_Esport.jsonl",
        output_path="data/interim/esport_review.csv",
        domain="esport"
    )

    remove_duplicates(
        input_path="data/interim/esport_review.csv",
        output_path="data/interim/clean_esport.csv",
    )