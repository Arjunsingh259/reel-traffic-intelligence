# System Architecture - Reel Traffic Intelligence

            ┌──────────────────────────────┐
            │   Traffic Capture (PCAP)     │
            └───────────┬──────────────────┘
                        │
            ┌───────────▼──────────────────┐
            │ Feature Extraction (Python)  │
            │ - Packet size & intervals    │
            │ - Burstiness & jitter        │
            │ - Throughput & QoS metrics  │
            └───────────┬──────────────────┘
                        │
            ┌───────────▼──────────────────┐
            │    AI Model (ML/DL)          │
            │ - Random Forest / CNN+LSTM   │
            │ - Predict reel/non-reel      │
            └───────────┬──────────────────┘
                        │
      ┌─────────────────▼────────────────────┐
      │ Real-Time Adaptation (UE Simulator) │
      │ - Prioritize reel traffic            │
      │ - Optimize bandwidth & prefetching  │
      └─────────────────────────────────────┘
