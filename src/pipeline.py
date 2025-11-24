import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnTransformerWrapper(BaseEstimator, TransformerMixin):
    """Wrap a fitted ColumnTransformer so it can be used as a sklearn transformer
    in a Pipeline and safely transform raw DataFrames into the numeric matrix
    that the trained model expects.

    Behavior:
    - Drops an `orig_index` column if present (it wasn't present when fitted).
    - Adds a temporary `Class` column when missing so passthrough remainder
      ordering matches the original fit; removes it after transform.
    - Removes any transformed remainder `Class` column from the transform
      output so the resulting array has the same number of features the
      model was trained on.
    """

    def __init__(self, column_transformer):
        self.ct = column_transformer
        # gather output names if available
        try:
            self.out_names_ = list(self.ct.get_feature_names_out())
        except Exception:
            self.out_names_ = None
        # locate the remainder Class if present in out_names_
        self._remainder_idx = None
        if self.out_names_ is not None:
            for i, n in enumerate(self.out_names_):
                if str(n).endswith('__Class') or str(n) == 'remainder__Class':
                    self._remainder_idx = i
                    break

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Accept either DataFrame or numpy array. If array, pass through ct.
        if not isinstance(X, (pd.DataFrame,)):
            arr = self.ct.transform(X)
            if self._remainder_idx is not None:
                arr = np.delete(arr, self._remainder_idx, axis=1)
            return arr

        Xc = X.copy()
        # Drop transient index column if present
        if 'orig_index' in Xc.columns:
            Xc = Xc.drop(columns=['orig_index'])

        class_was_missing = False
        if 'Class' not in Xc.columns:
            # add dummy Class so remainder passthrough matches fitted input
            Xc['Class'] = 0
            class_was_missing = True

        arr = self.ct.transform(Xc)

        # Reconstruct output column list like src/preprocess.py
        vcols = [c for c in Xc.columns if c.startswith('V')]
        transformers = []
        if vcols:
            transformers.append(('v_std', None, vcols))
        if 'Time' in Xc.columns:
            transformers.append(('time_std', None, ['Time']))
        if 'Amount' in Xc.columns:
            transformers.append(('amt_robust', None, ['Amount']))

        trans_cols = sum([cols for _, _, cols in transformers], [])
        remainder_cols = [c for c in Xc.columns if c not in trans_cols]
        out_cols = []
        for _, _, cols in transformers:
            out_cols.extend(cols)
        out_cols.extend(remainder_cols)

        X_scaled_df = pd.DataFrame(arr, columns=out_cols)

        # Reorder to original DataFrame order
        cols_order = list(Xc.columns)
        X_scaled_df = X_scaled_df[cols_order]

        # Drop dummy Class if we added it
        if class_was_missing:
            X_scaled_df = X_scaled_df.drop(columns=['Class'])

        feature_cols = [c for c in X_scaled_df.columns if c not in ('Class',)]
        return X_scaled_df[feature_cols].values
