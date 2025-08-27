from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import numpy as np

FEATURES: List[str] = [
    "session_duration_sec",
    "likes",
    "comments",
    "shares",
    "scroll_speed_items_per_sec",
    "watch_time_pct",
    "switch_rate_per_min",
    "interactions_per_min",
    "video_length_sec",
    "is_reel_surface",
    "autoplay",
    "pause_count",
    "seek_count",
]

DTYPES: Dict[str, Any] = {
    "session_duration_sec": float,
    "likes": int,
    "comments": int,
    "shares": int,
    "scroll_speed_items_per_sec": float,
    "watch_time_pct": float,
    "switch_rate_per_min": float,
    "interactions_per_min": float,
    "video_length_sec": float,
    "is_reel_surface": int,
    "autoplay": int,
    "pause_count": int,
    "seek_count": int,
}

DEFAULTS: Dict[str, Any] = {
    "session_duration_sec": 0.0,
    "likes": 0,
    "comments": 0,
    "shares": 0,
    "scroll_speed_items_per_sec": 0.0,
    "watch_time_pct": 0.0,
    "switch_rate_per_min": 0.0,
    "interactions_per_min": 0.0,
    "video_length_sec": 0.0,
    "is_reel_surface": 0,
    "autoplay": 0,
    "pause_count": 0,
    "seek_count": 0,
}

def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = DEFAULTS[col]
    for col, typ in DTYPES.items():
        if col in df.columns:
            try:
                if typ is int:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULTS[col]).astype(int)
                elif typ is float:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULTS[col]).astype(float)
                else:
                    df[col] = df[col].astype(typ)
            except Exception:
                df[col] = DEFAULTS[col]
    df["watch_time_pct"] = df["watch_time_pct"].clip(0, 100)
    df["scroll_speed_items_per_sec"] = df["scroll_speed_items_per_sec"].clip(lower=0)
    df["switch_rate_per_min"] = df["switch_rate_per_min"].clip(lower=0)
    df["interactions_per_min"] = df["interactions_per_min"].clip(lower=0)
    df["video_length_sec"] = df["video_length_sec"].clip(lower=0)
    df["session_duration_sec"] = df["session_duration_sec"].clip(lower=0)
    df = df[FEATURES]
    return df
