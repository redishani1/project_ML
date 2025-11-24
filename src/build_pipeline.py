import joblib
import os
from sklearn.pipeline import Pipeline
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALER_PATH = ROOT / 'models' / 'scalers_dedup.joblib'
MODEL_PATH = ROOT / 'models' / 'xgb_tuned_sigmoid.joblib'
OUT_PATH = ROOT / 'models' / 'final_pipeline_prod.joblib'


def build_pipeline(scaler_path=SCALER_PATH, model_path=MODEL_PATH, out_path=OUT_PATH):
    print('Loading scaler from', scaler_path)
    scaler = joblib.load(scaler_path)
    print('Loading model from', model_path)
    model = joblib.load(model_path)

    # Import the wrapper from the src package so the class is defined in an
    # importable module (this ensures it can be pickled by joblib).
    from src.pipeline import ColumnTransformerWrapper

    pre = ColumnTransformerWrapper(scaler)
    pipe = Pipeline([('preprocessor', pre), ('model', model)])

    print('Saving pipeline to', out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    joblib.dump(pipe, out_path)
    print('Saved final pipeline to', out_path)


if __name__ == '__main__':
    build_pipeline()
