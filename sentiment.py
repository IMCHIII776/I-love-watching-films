"""Shared text preparation, data validation, and neural-network architecture."""
import csv
import hashlib
import html
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(text)))
    return re.sub(r"\s+", " ", text).strip().lower()


def read_reviews(path):
    """Read text and label only. Never use rating as an input feature."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must have text and label columns; rating is ignored.")
        rows = list(reader)
    seen, texts, labels = {}, [], []
    dropped_empty = dropped_duplicates = 0
    mapping = {"0": 0, "1": 1, "negative": 0, "positive": 1}
    for line, row in enumerate(rows, start=2):
        label = str(row["label"]).strip().lower()
        if label not in mapping:
            raise ValueError(f"Invalid label on CSV line {line}: {label!r}")
        text = clean_text(row["text"] or "")
        if not text:
            dropped_empty += 1
            continue
        value = mapping[label]
        if text in seen:
            if seen[text] != value:
                raise ValueError("Identical normalized text has conflicting labels.")
            dropped_duplicates += 1
            continue
        seen[text] = value
        texts.append(text)
        labels.append(value)
    counts = Counter(labels)
    if set(counts) != {0, 1} or min(counts.values()) < 10:
        raise ValueError("Need at least 10 unique nonempty examples per class.")
    return texts, labels, {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(rows),
        "usable_rows": len(texts),
        "dropped_empty": dropped_empty,
        "dropped_duplicates": dropped_duplicates,
        "class_counts": {str(k): v for k, v in sorted(counts.items())},
        "input_columns": ["text"],
        "target_column": "label",
        "ignored_columns": ["rating"],
    }


class SentimentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs):
        return self.layers(inputs).squeeze(-1)
