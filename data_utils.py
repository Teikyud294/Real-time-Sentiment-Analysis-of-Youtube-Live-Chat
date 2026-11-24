"""Đọc dữ liệu, chuẩn hóa và tạo tập chia dùng chung cho hai mô hình."""

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import regex
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from src.preprocessing.dictionary import Dictionary
from src.preprocessing.normalization.spans_protector import PatternDetector

ROOT = Path(__file__).resolve().parent
DOMAINS = ("esport", "football", "music")
LABELS = (0, 1, 2)
LABEL_NAMES = ("positive", "negative", "neutral")
TEENCODE = {"lẹ": "nhanh", "zô": "vào", "zời": "trời", "hok": "không", "ko": "không",
            "khum": "không", "đc": "được", "mik": "mình", "j": "gì"}
EMOTE = r":(?=[\w+&’'-]*[^\W\d_])[\w+&’'-]+:"
PLACEHOLDER = r"<(?:url|user|num|entity|empty)>"
PROTECTED = re.compile(f"({EMOTE}|{PLACEHOLDER})")
MENTION = re.compile(r"(?<![\w@])@[\w.-]*[\w]")
NUMBER = re.compile(r"(?<!\w)\d+(?:\s*[-–:−]\s*\d+|[.,]\d+)?(?!\w)")
REPEATED = re.compile(r"([^\W\d_])\1{2,}")
WORD = re.compile(r"\b[^\W\d_]+\b")
GRAPHEME = regex.compile(r"\X")
EMOJI = regex.compile(r"\p{Emoji_Presentation}|\uFE0F|\u20E3")
EMOJI_WITH_TEXT_STYLE = regex.compile(r"\p{Emoji_Presentation}|\uFE0F|\u20E3|(?=\p{Extended_Pictographic})\p{Emoji}")
MISSING = {"", "nan", "<na>", "null"}


def configure_console():
    # Giữ được log tiếng Việt cả khi Windows chuyển hướng stdout/stderr vào file.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def basic_text(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def replace_emoji(text, emoji_mode):
    pattern = EMOJI if emoji_mode == "keep" else EMOJI_WITH_TEXT_STYLE
    if emoji_mode == "keep":
        replacement = lambda token: f" {token} "
    elif emoji_mode == "remove":
        replacement = lambda token: " "
    else:
        from emoji import demojize

        replacement = lambda token: f" {demojize(token)} "
    return GRAPHEME.sub(
        lambda match: replacement(match[0]) if pattern.search(match[0]) else match[0],
        text,
    )


@dataclass
class Preprocessor:
    emoji_mode: str = "keep"
    teencode: bool = False
    mask_numbers: bool = False
    entities: tuple[str, ...] = ()
    segment: bool = False
    normalize: bool = True
    _entity_pattern: re.Pattern = field(init=False, repr=False)

    def __post_init__(self):
        names = sorted({basic_text(name) for name in self.entities} - {""},
                       key=lambda name: (-len(name), name))
        self._entity_pattern = (
            re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)")
            if names else None
        )

    def __call__(self, text):
        text = basic_text(text) if self.normalize else " ".join(text.split())
        # Tách emote và placeholder trước khi sửa ký tự, số hay tách từ.
        parts = PROTECTED.split(text)
        for index in range(0, len(parts), 2):
            part = parts[index]
            if self.normalize:
                part = PatternDetector.URL.sub(
                    lambda match: " <url> " + match[0][len(match[0].rstrip(".,!?;:")):],
                    part,
                )
                part = MENTION.sub(" <user> ", part)
            part = replace_emoji(part, self.emoji_mode)
            chunks = PROTECTED.split(part)
            for offset in range(0, len(chunks), 2):
                chunk = chunks[offset]
                if self._entity_pattern:
                    chunk = self._entity_pattern.sub(" <entity> ", chunk)
                if self.normalize:
                    chunk = REPEATED.sub(r"\1\1", chunk)
                if self.mask_numbers:
                    chunk = NUMBER.sub(" <num> ", chunk)
                if self.teencode:
                    chunk = WORD.sub(lambda match: TEENCODE.get(match[0], match[0]), chunk)
                chunks[offset] = chunk
            parts[index] = " ".join(chunks)
        if self.emoji_mode == "remove":
            parts[1::2] = [" " if part.startswith(":") else part for part in parts[1::2]]
        result = " ".join(" ".join(parts).split())
        if self.segment and result:
            from pyvi import ViTokenizer

            chunks = PROTECTED.split(result)
            for index in range(0, len(chunks), 2):
                chunks[index] = ViTokenizer.tokenize(chunks[index].strip()) if chunks[index].strip() else ""
            result = " ".join(filter(None, chunks))
        return result or "<empty>"


def load_entities(dictionary_dir):
    dictionary = Dictionary(base_dir=dictionary_dir, auto_load=False)
    categories = {"PLAYER", "TEAM", "HERO", "ARTIST", "MUSIC_GROUP", "HOST_MC", "FANDOM"}
    names = set()
    for domain in ("esports", "football", "music"):
        entries = dictionary.load_csv(domain, dictionary.FILES[domain])
        for entry in entries:
            if entry.category in categories:
                names.update((*entry.surface_forms, entry.canonical))
    return tuple(sorted(names))


def add_common_args(parser):
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    for domain in DOMAINS:
        parser.add_argument(f"--{domain}-csv", type=Path, help=f"Ghi đè CSV của miền {domain}")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--dictionary-dir", type=Path, default=ROOT / "data/dictionaries")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/sentiment")
    parser.add_argument("--scenario", choices=("all", "in-domain", "cross-domain"), default="all")
    parser.add_argument("--held-out-domain", choices=DOMAINS, help="Chỉ chạy miền test này trong cross-domain")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force-kfold", action="store_true")
    parser.add_argument("--min-holdout-minority", type=int, default=5,
                        help="Số mẫu kỳ vọng tối thiểu của mỗi lớp trong val/test của holdout")
    parser.add_argument("--emoji-mode", choices=("keep", "demojize", "remove"), default="keep")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teencode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-numbers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-entities", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--class-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--variant", default="main", help="Tên cấu hình dùng để ghép kết quả ablation")


def make_preprocessor(args, segment=False):
    entities = load_entities(args.dictionary_dir) if args.mask_entities else ()
    return Preprocessor(args.emoji_mode, args.teencode, args.mask_numbers,
                        entities, segment, args.normalize)


def load_data(args):
    # Dùng một khóa cố định để các ablation có cùng tập chia.
    key_for = Preprocessor()
    frames = []
    for domain in DOMAINS:
        path = getattr(args, f"{domain}_csv") or args.data_dir / f"train_{domain}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", quotechar='"', dtype=str,
                            keep_default_na=False, usecols=[args.text_column, args.label_column])
        frame.columns = frame.columns.map({args.text_column: "text", args.label_column: "label"})
        frame = frame[~frame["text"].map(basic_text).isin(MISSING)
                      & ~frame["label"].map(basic_text).isin(MISSING)].copy()
        label = frame["label"].str.strip()
        invalid = ~label.isin(map(str, LABELS))
        if invalid.any():
            raise ValueError(f"{path}: nhãn ngoài 0, 1, 2: {label[invalid].unique().tolist()}")
        frame["label"] = label.astype(int)
        frame["domain"] = domain
        frame["key"] = frame["text"].map(key_for)
        logging.info("Nguồn %s | %d dòng hợp lệ | %s", path, len(frame),
                     frame["label"].value_counts().sort_index().to_dict())
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    conflicts = data.groupby("key")["label"].nunique()
    if conflicts.gt(1).any():
        logging.warning("Có %d nhóm nội dung trùng nhưng khác nhãn. Giữ mỗi nhãn và chia theo nhóm.",
                        conflicts.gt(1).sum())
    return data.drop_duplicates(["domain", "key", "label"])


def cv_indices(data, n_splits, seed):
    group_support = data.groupby("label")["key"].nunique().reindex(LABELS, fill_value=0)
    if group_support.min() < n_splits:
        raise ValueError(f"Cần ít nhất {n_splits} nhóm mỗi lớp để chạy CV: {group_support.to_dict()}")
    if data["key"].is_unique:
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        return list(cv.split(data, data["label"]))
    # Thử các phép chia tái lập vì StratifiedGroupKFold phân bổ nhóm xấp xỉ.
    for offset in range(16):
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + offset)
        splits = list(cv.split(data, data["label"], data["key"]))
        if all(data.iloc[index]["label"].nunique() == len(LABELS)
               for split in splits for index in split):
            return splits
    raise ValueError("Không thể tạo các fold có đủ ba nhãn với các nhóm trùng hiện tại.")


class InsufficientGroups(ValueError):
    """Tập quá nhỏ để chia holdout đủ nhãn."""


def holdout(data, fraction, seed):
    # Đại diện các nhóm xung đột bằng nhãn hiếm nhất của nhóm.
    counts = data["label"].value_counts()
    groups = (data.assign(frequency=data["label"].map(counts))
              .sort_values(["frequency", "label"], kind="stable").drop_duplicates("key"))
    if groups["label"].value_counts().reindex(LABELS, fill_value=0).min() < 2:
        raise InsufficientGroups("Cần ít nhất hai nhóm đại diện cho mỗi lớp để chia train/validation.")
    try:
        _, selected = train_test_split(groups, test_size=fraction, random_state=seed,
                                       stratify=groups["label"])
    except ValueError:
        selected = groups.iloc[:0]
    if selected["label"].nunique() < len(LABELS) or (
        groups.loc[~groups["key"].isin(selected["key"]), "label"].nunique() < len(LABELS)
    ):
        # Ở fold hiếm, buộc mỗi lớp có ít nhất một nhóm ở cả hai phía.
        rng = np.random.default_rng(seed)
        selected = pd.concat([
            group.sample(n=np.clip(round(len(group) * fraction), 1, len(group) - 1),
                         random_state=rng)
            for _, group in groups.groupby("label")
        ])
    mask = data["key"].isin(selected["key"])
    return data.loc[~mask], data.loc[mask]


@dataclass
class Split:
    scenario: str
    held_out: str
    fold: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_splits(data, args):
    seed = args.split_seed
    if args.scenario in {"all", "in-domain"}:
        pooled = data.drop_duplicates(["key", "label"])
        support = pooled.groupby("label")["key"].nunique().reindex(LABELS, fill_value=0)
        use_cv = args.force_kfold or support.min() * 0.15 < args.min_holdout_minority
        if not use_cv:
            try:
                rest, test = holdout(pooled, 0.15, seed)
                train, val = holdout(rest, 0.15 / 0.85, seed + 1)
            except InsufficientGroups:
                use_cv = True
            else:
                yield Split("in-domain", "all", 0, train, val, test)
        if use_cv:
            logging.info("In-domain: dùng 5 fold do ngưỡng lớp hiếm hoặc --force-kfold.")
            for fold, (rest, test) in enumerate(cv_indices(pooled, 5, seed), 1):
                train, val = holdout(pooled.iloc[rest], 0.15 / 0.85, seed + fold)
                yield Split("in-domain", "all", fold, train, val, pooled.iloc[test])
    if args.scenario in {"all", "cross-domain"}:
        for domain in (args.held_out_domain,) if args.held_out_domain else DOMAINS:
            test = data.loc[data["domain"].eq(domain)]
            if test.empty:
                raise ValueError(f"Miền test {domain} không có dữ liệu hợp lệ.")
            source = data.loc[~data["domain"].eq(domain)]
            overlaps = source["key"].isin(test["key"])
            logging.info("Miền test %s: bỏ %d dòng nguồn có nội dung trùng với test.", domain, overlaps.sum())
            source = source.loc[~overlaps].drop_duplicates(["key", "label"])
            train, val = holdout(source, 0.15, seed)
            yield Split("cross-domain", domain, 0, train, val, test)
