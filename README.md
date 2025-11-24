# Credit Card Fraud Detection ML Pipeline

[![CI](https://github.com/redishani1/project_ML/actions/workflows/ci.yml/badge.svg)](https://github.com/redishani1/project_ML/actions/workflows/ci.yml)

End-to-end machine learning pipeline for credit card fraud detection using XGBoost with probability calibration, FastAPI inference server, and production-ready deployment tooling.

## Features

- **High-performance model**: XGBoost classifier optimized for AUPRC (Average Precision)
- **Probability calibration**: Platt scaling for reliable fraud probability estimates
- **Production pipeline**: Single-artifact deployment with preprocessing + model
- **Real-time API**: FastAPI server with input validation and OpenAPI docs
- **Batch inference**: CLI tool for offline scoring
- **Full reproducibility**: Complete training, tuning, and evaluation scripts

## Project Structure

```
project_ML/
├── src/
│   ├── preprocess.py          # Data preprocessing and scaling
│   ├── train_models.py         # Train baseline and ensemble models
│   ├── tune_xgb.py            # Hyperparameter tuning (RandomizedSearchCV)
│   ├── pipeline.py            # Production pipeline wrapper
│   ├── build_pipeline.py      # Build final deployment artifact
│   ├── infer.py               # Batch inference CLI
│   └── serve_example.py       # FastAPI real-time server
├── tests/
│   └── test_smoke.py          # Smoke tests for CI
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI pipeline
├── requirements.txt           # Python dependencies
└── README_DEPLOY.md          # Deployment quick-start guide
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/redishani1/project_ML.git
cd project_ML
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

Place your `creditcard.csv` dataset in the project root, then preprocess:

```bash
python src/preprocess.py
```

This creates normalized datasets and saves scalers to `models/scalers_dedup.joblib`.

### 3. Train and Tune Models

Train baseline models and tune XGBoost:

```bash
python src/train_models.py
python src/tune_xgb.py
```

Models and evaluation reports are saved to `models/` and `reports/`.

### 4. Build Production Pipeline

Create a single deployment artifact combining preprocessing and calibrated model:

```bash
python -m src.build_pipeline
```

This saves `models/final_pipeline_prod.joblib`.

### 5. Run Inference

**Batch inference (offline):**

```bash
python src/infer.py --input data/test.csv --raw --threshold 0.004
```

**Real-time API server:**

```bash
python -m uvicorn src.serve_example:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for interactive API documentation.

**Example API request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "Time": 0, "V1": 0.1, "V2": -0.2, "V3": 0.05, "V4": 0.0,
      "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0,
      "V10": 0.0, "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0,
      "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0,
      "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
      "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0, "Amount": 10
    }]
  }'
```

## Model Performance

- **Metric optimized**: AUPRC (Average Precision) - ideal for imbalanced datasets
- **Test set AP**: ~0.838 (calibrated model on held-out test set)
- **Calibration**: Platt (sigmoid) scaling on validation set
- **Operating threshold**: ~0.004 for precision ≥85%

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Fraud probability prediction
  - Input: JSON with `records` array containing transaction features
  - Output: JSON with `predictions` array of fraud probabilities (0-1)
- `GET /docs` - Interactive OpenAPI documentation

## Development

Run tests:
```bash
pytest
```

Run syntax check:
```bash
python -m compileall -q .
```

## Deployment

See [README_DEPLOY.md](README_DEPLOY.md) for detailed deployment instructions including:
- Running the FastAPI server
- Network configuration for LAN/remote access
- Example client requests

## Requirements

- Python 3.11+
- scikit-learn
- xgboost
- pandas
- numpy
- joblib
- fastapi
- uvicorn

See `requirements.txt` for complete dependency list.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Issues and pull requests are welcome! Please ensure tests pass before submitting.
