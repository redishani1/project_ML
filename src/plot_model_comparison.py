"""Produce combined model performance visualizations.

Loads saved models from `models/`, scores `data/test.csv`, and creates:
- `reports/combined_pr_curve.png` : PR curves for all models on one plot
- `reports/ap_bar_chart.png` : bar chart of Average Precision (AUPRC) per model
- `reports/model_performance_summary.csv` : table of AP and precision@k
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = r"C:/Documents/project_ML"
MODELS_DIR = os.path.join(ROOT, 'models')
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS, exist_ok=True)


def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    return float(y_true.iloc[topk].sum()) / float(len(topk))


def main():
    test_path = os.path.join(DATA_DIR, 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=['Class'])
    y_test = test['Class']

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')]
    results = []

    plt.figure(figsize=(8,6))
    for mf in sorted(model_files):
        path = os.path.join(MODELS_DIR, mf)
        try:
            m = joblib.load(path)
        except Exception as e:
            print('Could not load', mf, e)
            continue

        name = os.path.splitext(mf)[0]
        # Ensure columns ordering — assume model was trained on DataFrame with same columns
        try:
            if hasattr(m, 'predict_proba'):
                scores = m.predict_proba(X_test)[:,1]
            else:
                scores = m.decision_function(X_test)
        except Exception as e:
            print('Model prediction failed for', name, e)
            continue

        ap = average_precision_score(y_test, scores)
        prec, rec, _ = precision_recall_curve(y_test, scores)

        plt.plot(rec, prec, label=f'{name} (AP={ap:.3f})')

        # precision@k
        p50 = precision_at_k(y_test.reset_index(drop=True), scores, 50) if len(scores) >= 50 else np.nan
        p100 = precision_at_k(y_test.reset_index(drop=True), scores, 100) if len(scores) >= 100 else np.nan
        p500 = precision_at_k(y_test.reset_index(drop=True), scores, 500) if len(scores) >= 500 else np.nan

        results.append({'model': name, 'ap': ap, 'p_at_50': p50, 'p_at_100': p100, 'p_at_500': p500})

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves — All Models')
    plt.legend(fontsize='small')
    plt.tight_layout()
    out_pr = os.path.join(REPORTS, 'combined_pr_curve.png')
    plt.savefig(out_pr)
    plt.close()

    # Bar chart of APs
    resdf = pd.DataFrame(results).sort_values('ap', ascending=False)
    if resdf.empty:
        print('No results to plot')
        return
    plt.figure(figsize=(10,4))
    plt.bar(resdf['model'], resdf['ap'], color='C0')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Precision (AUPRC)')
    plt.title('Model AUPRC Comparison')
    plt.tight_layout()
    out_bar = os.path.join(REPORTS, 'ap_bar_chart.png')
    plt.savefig(out_bar)
    plt.close()

    # Save CSV summary
    resdf.to_csv(os.path.join(REPORTS, 'model_performance_summary.csv'), index=False)

    print('Saved:', out_pr)
    print('Saved:', out_bar)
    print('Saved:', os.path.join(REPORTS, 'model_performance_summary.csv'))


if __name__ == '__main__':
    main()
