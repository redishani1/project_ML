"""Hyperparameter tuning for XGBoost using RandomizedSearchCV optimizing AUPRC.

Saves:
- `reports/xgb_random_search_results.csv` (CV results summary)
- `models/xgb_tuned.joblib` (best estimator)
- `reports/xgb_tuned_metrics.csv` (test metrics)
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

ROOT = r"C:/Documents/project_ML"
DATA = os.path.join(ROOT, 'data')
REPORTS = os.path.join(ROOT, 'reports')
MODELS = os.path.join(ROOT, 'models')
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


def main(n_iter=30, cv_folds=3, random_state=42):
    train = pd.read_csv(os.path.join(DATA, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA, 'test.csv'))
    # drop helper index if present
    for df in [train, test]:
        if 'orig_index' in df.columns:
            df.drop(columns=['orig_index'], inplace=True)

    X = train.drop(columns=['Class'])
    y = train['Class']

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    model = XGBClassifier(use_label_encoder=False, eval_metric='aucpr', verbosity=0, n_jobs=4, random_state=random_state)

    param_dist = {
        'n_estimators': [100, 200, 400, 800],
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 3, 5, 10]
    }

    # scorer: use sklearn's built-in average_precision scorer (AUPRC)
    # Using the string name avoids passing unexpected kwargs to the
    # underlying function during cross-validation.
    scorer = 'average_precision'

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=4,
        verbose=2,
        return_train_score=False
    )

    # set constant scale_pos_weight on estimator
    search.estimator.set_params(scale_pos_weight=scale_pos_weight)

    print('Starting RandomizedSearchCV (this may take a while)...')
    search.fit(X, y)

    print('Best params:', search.best_params_)
    print('Best CV score (AP):', search.best_score_)

    # Save CV results
    cvres = pd.DataFrame(search.cv_results_)
    cvres.to_csv(os.path.join(REPORTS, 'xgb_random_search_results.csv'), index=False)

    # Best estimator -> evaluate on test
    best = search.best_estimator_
    joblib.dump(best, os.path.join(MODELS, 'xgb_tuned.joblib'))

    test_df = test
    X_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']
    y_scores = best.predict_proba(X_test)[:,1]
    ap_test = average_precision_score(y_test, y_scores)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f'XGB tuned (AP={ap_test:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve - XGBoost Tuned')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, 'pr_xgb_tuned.png'))
    plt.close()

    # Save metrics
    metrics = {'model': 'xgb_tuned', 'ap_test': ap_test}
    pd.DataFrame([metrics]).to_csv(os.path.join(REPORTS, 'xgb_tuned_metrics.csv'), index=False)

    print('Saved tuned model, CV results, and test PR curve/metrics.')


if __name__ == '__main__':
    main()
