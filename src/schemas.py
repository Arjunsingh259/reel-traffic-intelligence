
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List

class TrafficEvent(BaseModel):
    session_duration_sec: float = Field(..., ge=0)
    likes: int = 0
    comments: int = 0
    shares: int = 0
    scroll_speed_items_per_sec: float = 0.0
    watch_time_pct: float = Field(0.0, ge=0, le=100)
    switch_rate_per_min: float = 0.0
    interactions_per_min: float = 0.0
    video_length_sec: float = 0.0
    is_reel_surface: int = 0
    autoplay: int = 0
    pause_count: int = 0
    seek_count: int = 0

class PredictResponse(BaseModel):
    label: str
    probability: float
    features: List[float]

class BatchRequest(BaseModel):
    items: List[TrafficEvent]
