"""Small file helpers shared by the stages: folders, csv read/write and run naming."""

import csv
import os
import sys


def ensure(directory: str) -> str:
    """Create `directory` if missing and return it. For output paths."""
    os.makedirs(directory, exist_ok=True)
    return directory


def write_csv(path, rows, fieldnames):
    """Write a list of dicts to `path` as csv with the given column order, creating the folder if needed."""
    ensure(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    """Read a csv written by an earlier stage into a list of dicts."""
    if not os.path.isfile(path):
        sys.exit(f"error: {path} not found; run the earlier stage (e.g. prepare) first")
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_name(model_key, no_instruction):
    """Name a run after its model, with a __noinstr suffix for the no-instruction ablation."""
    return f"{model_key}__noinstr" if no_instruction else model_key
