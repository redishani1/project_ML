"""Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost.

Produces:
- `reports/compare_pr_curves_four_models.png`
- `reports/compare_ap_p100_four_models.png`
- `reports/compare_four_models_metrics.csv`
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = r"C:/Documents/project_ML"
MODELS = os.path.join(ROOT, 'models')
DATA = os.path.join(ROOT, 'data')
REPORTS = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS, exist_ok=True)

model_names = ['original_logreg', 'original_dt', 'original_rf', 'original_xgb']


def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    return float(y_true.iloc[topk].sum()) / float(len(topk))


def load_test():
    path = os.path.join(DATA, 'test.csv')
    df = pd.read_csv(path)
    if 'orig_index' in df.columns:
        df = df.drop(columns=['orig_index'])
    X = df.drop(columns=['Class'])
    y = df['Class'].reset_index(drop=True)
    return X, y


def main():
    X_test, y_test = load_test()

    metrics = []

    plt.figure(figsize=(8,6))
    for name in model_names:
        path = os.path.join(MODELS, name + '.joblib')
        if not os.path.exists(path):
            print('Model not found, skipping:', path)
            continue
        m = joblib.load(path)
        # compute scores
        try:
            if hasattr(m, 'predict_proba'):
                scores = m.predict_proba(X_test)[:,1]
            else:
                scores = m.decision_function(X_test)
                smin, smax = scores.min(), scores.max()
                if smax > smin:
                    scores = (scores - smin) / (smax - smin)
        except Exception as e:
            print('Prediction failed for', name, e)
            continue

        ap = average_precision_score(y_test, scores)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        p100 = precision_at_k(y_test, scores, 100) if len(scores) >= 100 else np.nan

        plt.plot(rec, prec, label=f'{name} (AP={ap:.3f})')

        metrics.append({'model': name, 'ap': ap, 'p_at_100': p100})

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves — Four Models')
    plt.legend(fontsize='small')
    plt.tight_layout()
    out1 = os.path.join(REPORTS, 'compare_pr_curves_four_models.png')
    plt.savefig(out1)
    plt.close()

    # Bar chart AP and p@100
    dfm = pd.DataFrame(metrics).set_index('model')
    if dfm.empty:
        print('No metrics computed; exiting')
        return

    fig, ax1 = plt.subplots(figsize=(8,4))
    dfm['ap'].plot(kind='bar', color='C0', ax=ax1, position=0, width=0.4)
    ax1.set_ylabel('Average Precision (AUPRC)')

    ax2 = ax1.twinx()
    dfm['p_at_100'].plot(kind='bar', color='C1', ax=ax2, position=1, width=0.4)
    ax2.set_ylabel('Precision@100')

    ax1.set_xticklabels(dfm.index, rotation=45, ha='right')
    plt.title('AP and Precision@100 — Four Models')
    plt.tight_layout()
    out2 = os.path.join(REPORTS, 'compare_ap_p100_four_models.png')
    plt.savefig(out2)
    plt.close()

    dfm.to_csv(os.path.join(REPORTS, 'compare_four_models_metrics.csv'))

    print('Saved:', out1)
    print('Saved:', out2)
    print('Saved:', os.path.join(REPORTS, 'compare_four_models_metrics.csv'))


if __name__ == '__main__':
    main()
