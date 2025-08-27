from __future__ import annotations
import json
from typing import List
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TrafficEvent, PredictResponse, BatchRequest
from .common import load_artifacts, predict_from_records

app = FastAPI(title="Reel vs Non-Reel Traffic Classifier", version="1.0.0")

# Allow all origins for demo; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, feat_cfg, label_map = load_artifacts()

def label_to_str(y: int) -> str:
    return "reel" if int(y) == 1 else "non_reel"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(event: TrafficEvent):
    rec = event.model_dump()
    yhat, proba, df = predict_from_records(model, [rec])
    label = label_to_str(int(yhat[0]))
    return PredictResponse(label=label, probability=float(proba[0]), features=df.iloc[0].tolist())

@app.post("/predict_batch")
def predict_batch(req: List[TrafficEvent] | BatchRequest):
    items = req if isinstance(req, list) else req.items
    records = [it.model_dump() for it in items]
    yhat, proba, df = predict_from_records(model, records)
    out = [
        {"label": label_to_str(int(yhat[i])), "probability": float(proba[i]), "features": df.iloc[i].tolist()}
        for i in range(len(records))
    ]
    return {"items": out}

@app.websocket("/ws")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            event = json.loads(data)
            yhat, proba, df = predict_from_records(model, [event])
            resp = {
                "label": label_to_str(int(yhat[0])),
                "probability": float(proba[0]),
                "features": df.iloc[0].tolist()
            }
            await websocket.send_text(json.dumps(resp))
    except Exception:
        await websocket.close()
