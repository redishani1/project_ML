Deployment & quick start

1) Build the production pipeline (one-time):

```bash
python -m src.build_pipeline
```

This will create `models/final_pipeline_prod.joblib` which contains the preprocessing wrapper
and the calibrated model.

2) Run the example FastAPI server (development):

```bash
# from workspace root
python -m uvicorn src.serve_example:app --host 0.0.0.0 --port 8000
```

3) Example prediction (curl):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d \
'{"records": [{"Time": 0, "V1": 0.1, "V2": -0.2, "V3": 0.05, "V4": 0.0, "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0, "Amount": 20}]}'
```

Notes:
- The server expects raw feature dictionaries (same column names as original CSV, except `orig_index` may be omitted).
- For production, build a small wrapper to validate inputs and run inside a proper ASGI server with TLS and auth.
