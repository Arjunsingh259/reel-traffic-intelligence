from __future__ import annotations
import argparse, asyncio, json
import websockets
import requests
from .data_simulator import simulate_dataset
from .features import FEATURES

def sample_event(reel_bias: float = 0.5) -> dict:
    df = simulate_dataset(n_rows=1, reel_ratio=reel_bias)
    row = df.iloc[0]
    return {f: float(row[f]) if f not in ("is_reel_surface","autoplay","likes","comments","shares","pause_count","seek_count") else int(row[f]) for f in FEATURES}

def send_rest(n: int, url: str):
    for i in range(n):
        ev = sample_event(0.5)
        r = requests.post(url, json=ev, timeout=10)
        print(i+1, r.json())

async def send_ws(n: int, url: str):
    async with websockets.connect(url) as ws:
        for i in range(n):
            ev = sample_event(0.5)
            await ws.send(json.dumps(ev))
            msg = await ws.recv()
            print(i+1, msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--ws", action="store_true", help="Use websocket")
    parser.add_argument("--rest_url", type=str, default="http://localhost:8000/predict")
    parser.add_argument("--ws_url", type=str, default="ws://localhost:8000/ws")
    args = parser.parse_args()

    if args.ws:
        asyncio.run(send_ws(args.n, args.ws_url))
    else:
        send_rest(args.n, args.rest_url)
