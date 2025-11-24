from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, ValidationError, Field
from typing import List
import joblib
import uvicorn
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PIPELINE_PATH = os.path.join(ROOT, 'models', 'final_pipeline_prod.joblib')

app = FastAPI(title='Credit Card Fraud Inference')


class Record(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float = Field(..., alias='V10')
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictRequest(BaseModel):
    records: List[Record]


class PredictionsResponse(BaseModel):
    predictions: List[float]


@app.on_event('startup')
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        raise RuntimeError(f'Pipeline not found at {PIPELINE_PATH}, run src/build_pipeline.py first')
    global pipeline
    pipeline = joblib.load(PIPELINE_PATH)


@app.get('/health')
def health():
    return {'status': 'ok'}


example_record = {
    "Time": 0,
    "V1": 0.1,
    "V2": -0.2,
    "V3": 0.05,
    "V4": 0.0,
    "V5": 0.0,
    "V6": 0.0,
    "V7": 0.0,
    "V8": 0.0,
    "V9": 0.0,
    "V10": 0.0,
    "V11": 0.0,
    "V12": 0.0,
    "V13": 0.0,
    "V14": 0.0,
    "V15": 0.0,
    "V16": 0.0,
    "V17": 0.0,
    "V18": 0.0,
    "V19": 0.0,
    "V20": 0.0,
    "V21": 0.0,
    "V22": 0.0,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": 0.0,
    "V27": 0.0,
    "V28": 0.0,
    "Amount": 10
}


@app.post(
    '/predict',
    response_model=PredictionsResponse,
    responses={200: {"content": {"application/json": {"example": {"predictions": [0.0003720294769395058]}}}}},
)
def predict(
    req: PredictRequest = Body(..., examples={
        "single": {
            "summary": "Single record",
            "value": {"records": [example_record]}
        },
        "multiple": {
            "summary": "Multiple records",
            "value": {"records": [example_record, example_record]}
        }
    })
):
    # Pydantic ensures each record contains required numeric fields; convert
    # to DataFrame in the original column order.
    try:
        df = pd.DataFrame([r.model_dump() for r in req.records])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Ensure columns are ordered as in the original dataset (no orig_index/class)
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing required fields: {missing}')

    X = df[cols]
    try:
        probs = pipeline.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error during prediction: {e}')
    return PredictionsResponse(predictions=probs)


if __name__ == '__main__':
    uvicorn.run('src.serve_example:app', host='0.0.0.0', port=8000, reload=True)
