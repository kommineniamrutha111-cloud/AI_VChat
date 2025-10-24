"""Filesystem helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_parents(path: Path) -> None:
    """Ensure parent directories exist for provided path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_temp_files(paths: Iterable[Path]) -> None:
    """Best-effort cleanup for temporary files."""
    for item in paths:
        try:
            if item.exists():
                item.unlink()
        except OSError:
            # Ignore cleanup errors but log in production implementations.
            continue

