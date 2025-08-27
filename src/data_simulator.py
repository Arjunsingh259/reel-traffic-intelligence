from __future__ import annotations
import numpy as np
import pandas as pd
from .features import FEATURES

rng = np.random.default_rng(42)

def _simulate_reel(n: int) -> pd.DataFrame:
    session_duration_sec = rng.normal(90, 35, n).clip(10, 360)
    likes = rng.poisson(3.5, n)
    comments = rng.poisson(0.8, n)
    shares = rng.poisson(0.6, n)
    scroll_speed_items_per_sec = rng.normal(1.5, 0.4, n).clip(0.3, 3.0)
    watch_time_pct = rng.normal(72, 15, n).clip(10, 100)
    switch_rate_per_min = rng.normal(3.5, 1.2, n).clip(0.2, 8.0)
    interactions_per_min = rng.normal(4.5, 1.5, n).clip(0.0, 12.0)
    video_length_sec = rng.normal(28, 10, n).clip(5, 90)
    is_reel_surface = np.ones(n, dtype=int)
    autoplay = rng.integers(0, 2, n, endpoint=False)
    pause_count = rng.poisson(0.3, n)
    seek_count = rng.poisson(0.2, n)

    return pd.DataFrame({
        "session_duration_sec": session_duration_sec,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "scroll_speed_items_per_sec": scroll_speed_items_per_sec,
        "watch_time_pct": watch_time_pct,
        "switch_rate_per_min": switch_rate_per_min,
        "interactions_per_min": interactions_per_min,
        "video_length_sec": video_length_sec,
        "is_reel_surface": is_reel_surface,
        "autoplay": autoplay,
        "pause_count": pause_count,
        "seek_count": seek_count,
    })

def _simulate_nonreel(n: int) -> pd.DataFrame:
    session_duration_sec = rng.normal(300, 120, n).clip(20, 2400)
    likes = rng.poisson(0.8, n)
    comments = rng.poisson(0.3, n)
    shares = rng.poisson(0.1, n)
    scroll_speed_items_per_sec = rng.normal(0.5, 0.2, n).clip(0.0, 2.0)
    watch_time_pct = rng.normal(30, 25, n).clip(0, 100)
    switch_rate_per_min = rng.normal(0.8, 0.6, n).clip(0.0, 4.0)
    interactions_per_min = rng.normal(1.0, 0.8, n).clip(0.0, 6.0)
    video_length_sec = rng.normal(240, 120, n).clip(0, 3600)
    is_reel_surface = np.zeros(n, dtype=int)
    autoplay = rng.integers(0, 2, n, endpoint=False)
    pause_count = rng.poisson(1.3, n)
    seek_count = rng.poisson(1.1, n)

    return pd.DataFrame({
        "session_duration_sec": session_duration_sec,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "scroll_speed_items_per_sec": scroll_speed_items_per_sec,
        "watch_time_pct": watch_time_pct,
        "switch_rate_per_min": switch_rate_per_min,
        "interactions_per_min": interactions_per_min,
        "video_length_sec": video_length_sec,
        "is_reel_surface": is_reel_surface,
        "autoplay": autoplay,
        "pause_count": pause_count,
        "seek_count": seek_count,
    })

def simulate_dataset(n_rows: int = 10000, reel_ratio: float = 0.5) -> pd.DataFrame:
    n_reel = int(n_rows * reel_ratio)
    n_non = n_rows - n_reel

    df_reel = _simulate_reel(n_reel)
    df_non = _simulate_nonreel(n_non)

    df_reel["label"] = 1
    df_non["label"] = 0

    df = pd.concat([df_reel, df_non], axis=0, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df
