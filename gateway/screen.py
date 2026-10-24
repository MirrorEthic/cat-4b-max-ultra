"""The rules layer outside the model (docs/GATEWAY.md section 4). Reloads screen_rules.yaml when it changes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

RULES_FILE = Path(__file__).resolve().parent / "screen_rules.yaml"


@dataclass
class Screen:
    path: Path = RULES_FILE
    _mtime: float = 0.0
    _rules: list[tuple[str, str, list[re.Pattern]]] = field(default_factory=list)

    def _load(self):
        mtime = self.path.stat().st_mtime
        if mtime == self._mtime and self._rules:
            return
        data = yaml.safe_load(self.path.read_text())
        rules = []
        for name, cat in data["categories"].items():
            pats = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in cat["patterns"]]
            rules.append((name, cat["redirect"].strip(), pats))
        self._rules, self._mtime = rules, mtime

    def check(self, text: str) -> tuple[str, str] | None:
        """(category, redirect) for the first category with a matching pattern, else None."""
        self._load()
        flat = " ".join(text.split())
        for name, redirect, pats in self._rules:
            if any(p.search(flat) for p in pats):
                return name, redirect
        return None

    def categories(self) -> list[str]:
        self._load()
        return [r[0] for r in self._rules]
