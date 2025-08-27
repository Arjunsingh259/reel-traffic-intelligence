# Technical Documentation - Reel Traffic Intelligence

## Overview
Reel Traffic Intelligence detects and classifies real-time reel/video vs non-reel traffic in social networking applications. It enables user equipment (UE) to optimize performance dynamically under network congestion and varying coverage conditions.

## System Design
- **Traffic Capture:** Packet sniffing using Python (scapy) or pre-recorded datasets.
- **Feature Extraction:** Extracts session, packet, network, and temporal features.
- **Model:** Hybrid CNN + BiLSTM or Random Forest baseline.
- **Inference:** Real-time prediction with probability and uncertainty estimation.
- **Action Layer:** Adaptive bandwidth and prefetching based on traffic classification.

## Features Used
- Session-level: session_duration, likes, comments, shares, scroll_speed, interactions.
- Video-level: video_length, autoplay, pause_count, seek_count.
- Network-level: throughput, RTT, packet loss, signal strength, network type.
- Temporal: sliding window summaries, event sequences.

## Dataset
- **Public datasets:** MAWI, YouTube QoE.
- **Synthetic dataset:** Generated to mimic real network conditions.
- Data labeled as reel vs non-reel.
- Dataset stratified by network quality (good/moderate/poor).

## Model Architecture
- Input: Feature vector per event.
- Layers: Dense → 1D CNN → BiLSTM → Dense → Sigmoid output.
- Output: Probability of reel traffic.
- Optional: Monte-Carlo Dropout for uncertainty estimation.

## Training
- Loss: Binary cross-entropy.
- Optimizer: Adam.
- Stratified split by label and network quality.
- Calibration: Platt scaling for output probability.
- Evaluation: ROC-AUC, F1, Precision, Recall.

## Deployment
- Lightweight model for edge devices.
- Quantization to reduce memory footprint.
- Inference latency: <100ms per event.

## Ethical Considerations
- Only metadata used; no personal data collected.
- Open-source and reproducible.
