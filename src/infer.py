import os
import argparse
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MODEL = os.path.join(ROOT, 'models', 'xgb_tuned_sigmoid.joblib')
DEFAULT_SCALER = os.path.join(ROOT, 'models', 'scalers_dedup.joblib')


def load_data(path):
    df = pd.read_csv(path)
    return df


def main():
    p = argparse.ArgumentParser(description='Batch inference script')
    p.add_argument('--input', '-i', required=True, help='Input CSV file')
    p.add_argument('--output', '-o', default=None, help='Output CSV with scores (default: reports/inference_scores.csv)')
    p.add_argument('--model', default=DEFAULT_MODEL, help='Calibrated model joblib')
    p.add_argument('--scaler', default=DEFAULT_SCALER, help='Scaler joblib (for raw inputs)')
    p.add_argument('--raw', action='store_true', help='If set, treat input as raw (apply scaler). Otherwise expects normalized features')
    p.add_argument('--threshold', type=float, default=None, help='Probability threshold to binarize predictions')
    args = p.parse_args()

    df = load_data(args.input)
    model = joblib.load(args.model)

    if args.raw:
        if not os.path.exists(args.scaler):
            raise FileNotFoundError(args.scaler)
        scaler = joblib.load(args.scaler)

        # Work with a copy and remove any runtime-added index column that wasn't
        # present when the scaler was originally fit (e.g. 'orig_index'). The
        # scaler was fit on the original creditcard CSV which did not include
        # an extra index column.
        X = df.copy()
        if 'orig_index' in X.columns:
            X = X.drop(columns=['orig_index'])

        # If labels are absent (production inference), add a dummy 'Class'
        # column so the ColumnTransformer's remainder passthrough mapping
        # matches the object it was fit on. We'll drop it after transformation.
        class_was_missing = False
        if 'Class' not in X.columns:
            X['Class'] = 0
            class_was_missing = True

        # Reconstruct the output column order exactly like `src/preprocess.py` did
        # when it converted the transformer output back into a DataFrame.
        vcols = [c for c in X.columns if c.startswith('V')]
        transformers = []
        if vcols:
            transformers.append(('v_std', None, vcols))
        if 'Time' in X.columns:
            transformers.append(('time_std', None, ['Time']))
        if 'Amount' in X.columns:
            transformers.append(('amt_robust', None, ['Amount']))

        trans_cols = sum([cols for _, _, cols in transformers], [])
        remainder_cols = [c for c in X.columns if c not in trans_cols]
        out_cols = []
        for _, _, cols in transformers:
            out_cols.extend(cols)
        out_cols.extend(remainder_cols)

        # Apply the fitted ColumnTransformer to the DataFrame. The returned
        # array has columns in the order described by `out_cols`.
        X_trans = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_trans, columns=out_cols)

        # Reorder to the original DataFrame column order used during preprocess
        cols_order = list(X.columns)
        X_scaled_df = X_scaled_df[cols_order]

        # If we added a dummy Class earlier, drop it now before scoring.
        if class_was_missing:
            X_scaled_df = X_scaled_df.drop(columns=['Class'])

        # Drop any remaining non-feature columns (like original index) and pass
        # the 2D feature array to the model in the correct order.
        feature_cols = [c for c in X_scaled_df.columns if c not in ('Class',)]
        X_for_model = X_scaled_df[feature_cols].values
        probs = model.predict_proba(X_for_model)[:, 1]
    else:
        # expect normalized features present; drop label if present
        X = df.drop(columns=['Class']) if 'Class' in df.columns else df
        probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out['score'] = probs
    if args.threshold is not None:
        out['pred'] = (out['score'] >= args.threshold).astype(int)

    out_dir = os.path.join(ROOT, 'reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, 'inference_scores.csv')
    out.to_csv(out_path, index=False)

    # if label present, report metrics
    if 'Class' in df.columns:
        y = df['Class']
        ap = average_precision_score(y, probs)
        print('Average precision (AP):', ap)
        if args.threshold is not None:
            tp, fp, tn, fn = None, None, None, None
            tn, fp, fn, tp = confusion_matrix(y, out['pred']).ravel()
            print('Confusion at threshold', args.threshold, 'TP,FP,TN,FN =', tp, fp, tn, fn)

    print('Saved scores to', out_path)


if __name__ == '__main__':
    main()
