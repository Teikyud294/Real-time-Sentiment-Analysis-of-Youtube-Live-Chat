from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path
import unicodedata
import json
import csv


@dataclass(frozen=True)
class DictionaryEntry:
    surface_forms: tuple[str, ...]
    canonical: str
    category: str
    domains: tuple[str, ...] = ()


class DictionaryLoadError(ValueError):
    pass


class Dictionary:

    FILES = {
        "standard": "Standard.csv",
        "teencode": "Teencode.csv",
        "esports": "EsportAOV.csv",
        "football": "FootballVN.csv",
        "music": "MusicEXSH.csv",
        "emoji": "Emoji.csv",
    }

    def __init__(self, base_dir: str | Path | None = None, auto_load: bool = True) -> None:
        
        self.base_dir = (
            Path(base_dir)
            if base_dir
            else Path(__file__).resolve().parents[2] / "data" / "dictionaries"
        )

        self.entries: dict[str, tuple[DictionaryEntry, ...]] = {}
        self.by_surface: dict[str, dict[str, tuple[DictionaryEntry, ...]]] = {}
        self.standard: frozenset[str] = frozenset()
        self.lexicon: frozenset[str] = frozenset()

        if auto_load:
            self.load_all()

    # --------------------------------------------------- NORMALIZATION
    @staticmethod
    def _key(value: str) -> str:
        return unicodedata.normalize("NFC", value).casefold()

    @staticmethod
    def _parse_list(value: str, domains: bool = False) -> tuple[str, ...]:
        if not value or not value.strip():
            return ()

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parts = value.replace(",", "|") if domains else value
            parsed = parts.split("|")

        if isinstance(parsed, str):
            parsed = [parsed]

        if not isinstance(parsed, list):
            return ()

        return tuple(
            unicodedata.normalize("NFC", item.strip())
            for item in parsed
            if isinstance(item, str) and item.strip()
        )

    # --------------------------------------------------- LOAD CSV
    def load_csv(self, name: str, path: str | Path) -> tuple[DictionaryEntry, ...]:

        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path

        if not path.is_file():
            raise DictionaryLoadError(f"Dictionary not found: {path}")

        loaded = []

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])

            # STANDARD: chỉ có cột text
            if name == "standard":
                if "text" not in columns:
                    raise DictionaryLoadError(f"Missing 'text': {path}")

                for row in reader:
                    text = (row.get("text") or "").strip()
                    if text:
                        loaded.append(DictionaryEntry(
                            surface_forms=(text,),
                            canonical=text,
                            category="STANDARD",
                            domains=("GENERAL",),
                        ))

            else:
                required = {
                    "surface_forms", "canonical", "category", "domains"
                }

                if missing := required - columns:
                    raise DictionaryLoadError(
                        f"{path.name}: missing {sorted(missing)}"
                    )

                for row in reader:
                    forms = self._parse_list(row.get("surface_forms") or "")
                    canonical = (row.get("canonical") or "").strip()
                    category = (row.get("category") or "").strip()
                    domains = self._parse_list(
                        row.get("domains") or "", domains=True
                    )

                    if not forms or not canonical:
                        continue

                    loaded.append(DictionaryEntry(
                        surface_forms=forms,
                        canonical=unicodedata.normalize("NFC", canonical),
                        category=category.upper() or "ENTITY",
                        domains=tuple(d.upper() for d in domains),
                    ))

        self.entries[name] = tuple(loaded)
        return self.entries[name]

    # --------------------------------------------------- BUILD INDEXES
    def build_indexes(self) -> None:
        self.by_surface.clear()
        lexicon = set()

        for name, entries in self.entries.items():
            index: dict[str, list[DictionaryEntry]] = {}

            for entry in entries:
                lexicon.add(self._key(entry.canonical))

                for surface in entry.surface_forms:
                    key = self._key(surface)
                    index.setdefault(key, []).append(entry)
                    lexicon.add(key)

            self.by_surface[name] = {
                key: tuple(items)
                for key, items in index.items()
            }

        self.standard = frozenset(
            self._key(e.canonical)
            for e in self.entries.get("standard", ())
        )
        self.lexicon = frozenset(lexicon)

    # --------------------------------------------------- LOAD ALL
    def load_all(self, config: dict | None = None) -> None:
        paths = dict(self.FILES)

        if config:
            cfg = config.get("dictionaries", {})
            paths.update({
                key: value
                for key, value in cfg.items()
                if key in paths and isinstance(value, str)
            })
            paths.update({
                domain.lower(): path
                for domain, path in cfg.get("domains", {}).items()
                if domain.lower() in paths
            })

        self.entries.clear()

        for name, filename in paths.items():
            path = Path(filename)

            if not path.is_absolute():
                candidates = [
                    self.base_dir / path,
                    self.base_dir / path.name,
                    self.base_dir.parent.parent / path,
                ]
                path = next(
                    (p for p in candidates if p.is_file()),
                    candidates[0],
                )

            self.load_csv(name, path)

        self.build_indexes()

    # --------------------------------------------------- LOOKUP
    def entries_for(
        self,
        name: str,
        text: str,
    ) -> tuple[DictionaryEntry, ...]:
        return self.by_surface.get(name.casefold(), {}).get(
            self._key(text), ()
        )

    # --------------------------------------------------- RESOLVE
    def resolve_surface(
        self,
        text: str,
        domain: str,
        dictionary_names: Iterable[str] = (
            "teencode", "esports", "football", "music"
        ),
    ) -> DictionaryEntry | None:

        domain = domain.upper()
        candidates = []

        for name in dictionary_names:
            if name in {"esports", "football", "music"}:
                if name.upper() != domain:
                    continue

            for entry in self.entries_for(name, text):
                if entry.domains and not (
                    domain in entry.domains or "GENERAL" in entry.domains
                ):
                    continue

                candidates.append(entry)

        if not candidates:
            return None

        # Chỉ thay thế nếu các kết quả thống nhất canonical
        if len({self._key(e.canonical) for e in candidates}) == 1:
            return candidates[0]

        return None
