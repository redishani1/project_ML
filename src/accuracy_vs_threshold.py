"""Plot Accuracy vs Threshold for all saved models on the test set.

Produces:
- `reports/accuracy_vs_threshold.png`
- `reports/accuracy_vs_threshold.csv`
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

ROOT = r"C:/Documents/project_ML"
MODELS_DIR = os.path.join(ROOT, 'models')
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS, exist_ok=True)


def get_scores(model, X):
    # returns probability-like scores in [0,1]
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        # scale to 0-1
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            return (scores - smin) / (smax - smin)
        return np.zeros_like(scores)
    raise ValueError('Model has no probability or decision function')


def main():
    test_path = os.path.join(DATA_DIR, 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)
    df = pd.read_csv(test_path)
    # drop helper column if present
    if 'orig_index' in df.columns:
        df = df.drop(columns=['orig_index'])
    X_test = df.drop(columns=['Class'])
    y_test = df['Class'].reset_index(drop=True)

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')]
    thresholds = np.linspace(0, 1, 101)
    records = []

    plt.figure(figsize=(8,6))
    for mf in sorted(model_files):
        path = os.path.join(MODELS_DIR, mf)
        try:
            m = joblib.load(path)
        except Exception as e:
            print('Skipping', mf, 'load failed:', e)
            continue

        name = os.path.splitext(mf)[0]
        try:
            scores = get_scores(m, X_test)
        except Exception as e:
            print('Skipping', name, 'predict failed:', e)
            continue

        accs = []
        for t in thresholds:
            preds = (scores >= t).astype(int)
            accs.append(accuracy_score(y_test, preds))

        plt.plot(thresholds, accs, label=name)
        for th, ac in zip(thresholds, accs):
            records.append({'model': name, 'threshold': th, 'accuracy': ac})

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold for models (test set)')
    plt.legend(fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(REPORTS, 'accuracy_vs_threshold.png')
    plt.savefig(out_plot)
    plt.close()

    recdf = pd.DataFrame(records)
    recdf.to_csv(os.path.join(REPORTS, 'accuracy_vs_threshold.csv'), index=False)

    print('Saved:', out_plot)
    print('Saved:', os.path.join(REPORTS, 'accuracy_vs_threshold.csv'))


if __name__ == '__main__':
    main()
