"""
Data loading and preprocessing utilities for the Gen Z NLP project.
"""

import ast
import re
from pathlib import Path
from typing import List

import pandas as pd


def safe_parse_list(s: str) -> list:
    """Parse stringified Python list from CSV, handling edge cases."""
    s = str(s).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
    inner = s.strip("[] \n")
    parts = [p.strip().strip("'\"") for p in re.split(r",\s*", inner) if p.strip()]
    return parts


def parse_dialog_column(s: str) -> List[str]:
    """Extract individual utterances from the dialog column."""
    utterances = safe_parse_list(str(s).strip())
    return [str(u).strip() for u in utterances if str(u).strip()]


def parse_label_column(s: str) -> List[int]:
    """Parse act or emotion label lists like '[2 1 3 2]' or '[0,0,0,0]'."""
    s = str(s).strip().strip("[]")
    parts = re.split(r"[,\s]+", s)
    return [int(p) for p in parts if p.strip()]


def load_dailydialog_csv(path: Path) -> pd.DataFrame:
    """Load and parse a DailyDialog CSV file."""
    df = pd.read_csv(path)
    df["dialog_list"] = df["dialog"].apply(parse_dialog_column)
    df["act_list"] = df["act"].apply(parse_label_column)
    df["emotion_list"] = df["emotion"].apply(parse_label_column)
    return df


def flatten_dialogues(
    df: pd.DataFrame, context_size: int = 2
) -> pd.DataFrame:
    """Explode each dialogue into utterance-level rows with optional context window."""
    rows = []
    for dialog_idx, row in df.iterrows():
        utts = row["dialog_list"]
        acts = row["act_list"]
        emos = row["emotion_list"]
        n = min(len(utts), len(acts), len(emos))
        for turn_idx in range(n):
            ctx_parts = []
            for j in range(max(0, turn_idx - context_size), turn_idx):
                ctx_parts.append(utts[j])
            context = " [SEP] ".join(ctx_parts) if ctx_parts else ""
            rows.append({
                "dialog_id": dialog_idx,
                "turn_id": turn_idx,
                "utterance": utts[turn_idx],
                "context": context,
                "input_text": (
                    (context + " [SEP] " + utts[turn_idx]) if context else utts[turn_idx]
                ),
                "act_label": acts[turn_idx],
                "emotion_label": emos[turn_idx],
            })
    return pd.DataFrame(rows)
