"""Train multiple classifiers and evaluate on validation and test sets.

Trains: Logistic Regression, Decision Tree, Random Forest, XGBoost
Training variants: original train (use class_weight='balanced'), SMOTE balanced train
Evaluates: Average Precision (AUPRC), precision@k (top 50/100/500), saves PR curves and model artifacts.
"""
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

ROOT = r"C:/Documents/project_ML"
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS = os.path.join(ROOT, 'reports')
MODELS = os.path.join(ROOT, 'models')
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


def load_csv(path):
    df = pd.read_csv(path)
    # drop any helper columns we added earlier
    if 'orig_index' in df.columns:
        df = df.drop(columns=['orig_index'])
    return df


def precision_at_k(y_true, y_scores, k):
    # compute precision among top-k highest scoring samples
    order = np.argsort(y_scores)[::-1]
    topk = order[:k]
    return int(y_true.iloc[topk].sum()) / float(len(topk))


def train_and_eval(train_df, val_df, test_df, variant_name):
    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']
    X_val = val_df.drop(columns=['Class'])
    y_val = val_df['Class']
    X_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']

    results = []

    models = OrderedDict()
    # Logistic Regression
    models['logreg'] = LogisticRegression(max_iter=1000, solver='lbfgs')
    # Decision Tree
    models['dt'] = DecisionTreeClassifier(random_state=42)
    # Random Forest
    models['rf'] = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    # XGBoost if available
    if XGBClassifier is not None:
        models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, n_jobs=4, random_state=42)

    for name, clf in models.items():
        clf_name = f"{variant_name}_{name}"
        print('\nTraining', clf_name)

        # For original variant we may want to use class_weight
        if variant_name == 'original':
            if hasattr(clf, 'class_weight'):
                clf.set_params(class_weight='balanced')
            # For xgb use scale_pos_weight
            if name == 'xgb' and XGBClassifier is not None:
                # compute scale_pos_weight
                neg = (y_train == 0).sum()
                pos = (y_train == 1).sum()
                if pos > 0:
                    clf.set_params(scale_pos_weight=float(neg) / float(pos))

        # fit
        clf.fit(X_train, y_train)
        # save model
        joblib.dump(clf, os.path.join(MODELS, clf_name + '.joblib'))

        # predict probabilities
        if hasattr(clf, 'predict_proba'):
            y_val_scores = clf.predict_proba(X_val)[:, 1]
            y_test_scores = clf.predict_proba(X_test)[:, 1]
        else:
            # fallback to decision function
            y_val_scores = clf.decision_function(X_val)
            y_test_scores = clf.decision_function(X_test)

        # Metrics
        ap_val = average_precision_score(y_val, y_val_scores)
        ap_test = average_precision_score(y_test, y_test_scores)

        # Precision@k
        k_list = [50, 100, 500]
        p_at_k = {}
        for k in k_list:
            if len(y_test_scores) >= k:
                p_at_k[k] = precision_at_k(y_test.reset_index(drop=True), pd.Series(y_test_scores), k)
            else:
                p_at_k[k] = np.nan

        # Save PR curve plots
        try:
            import matplotlib.pyplot as plt
            precision, recall, _ = precision_recall_curve(y_test, y_test_scores)
            plt.figure(figsize=(6,4))
            plt.plot(recall, precision, label=f'{clf_name} (AP={ap_test:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR curve - {clf_name}')
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(REPORTS, f'pr_{clf_name}.png')
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print('Could not plot PR curve for', clf_name, e)

        results.append({
            'variant': variant_name,
            'model': name,
            'ap_val': ap_val,
            'ap_test': ap_test,
            'p_at_50': p_at_k[50],
            'p_at_100': p_at_k[100],
            'p_at_500': p_at_k[500],
            'model_path': os.path.join(MODELS, clf_name + '.joblib')
        })

    return results


def main():
    # load datasets
    train_orig = load_csv(os.path.join(DATA_DIR, 'train.csv'))
    train_smote = load_csv(os.path.join(DATA_DIR, 'train_smote.csv'))
    val = load_csv(os.path.join(DATA_DIR, 'val.csv'))
    test = load_csv(os.path.join(DATA_DIR, 'test.csv'))

    all_results = []

    # Train on original with class_weight handling
    all_results.extend(train_and_eval(train_orig, val, test, 'original'))

    # Train on SMOTE balanced data
    all_results.extend(train_and_eval(train_smote, val, test, 'smote'))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(REPORTS, 'model_comparison_results.csv'), index=False)
    print('\nSaved results to', os.path.join(REPORTS, 'model_comparison_results.csv'))


if __name__ == '__main__':
    main()
