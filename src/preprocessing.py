# src/preprocessing.py
"""
Preprocessing pipeline for Titanic ML project.

- Applies feature engineering (src.features.create_features)
- Imputes Age using median per Title (custom transformer) on the full DataFrame
- Imputes numeric missing values (median) and scales numeric features
- Imputes categorical missing values and OneHot-encodes categoricals (version-safe)
- Exposes `get_preprocessor()` and `get_full_pipeline()` for training code to use

Run quick check:
    python3 -m src.preprocessing
"""

from typing import List
import os

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.features import create_features


# -------------------------
# Module-level wrapper for FE (pickleable)
# -------------------------
def _apply_create_features(X):
    """
    Wrapper used by FunctionTransformer in pipeline so the function is
    a proper module-level callable (pickleable).
    X will be a 2D numpy array or DataFrame depending on how pipeline is used.
    We call create_features which expects a pandas DataFrame and returns DataFrame.
    """
    # If X is already a DataFrame (when called in tests) use it directly; else convert.
    if isinstance(X, pd.DataFrame):
        df_in = X
    else:
        # if X is numpy array, try to recover column names if present on pandas input
        # best-effort conversion: convert to DataFrame without column names
        df_in = pd.DataFrame(X)
    return create_features(df_in)


# -------------------------
# Custom Transformers
# -------------------------
class AgeImputer(BaseEstimator, TransformerMixin):
    """
    Impute Age using median grouped by Title. Works on pandas DataFrame and returns DataFrame.
    This transformer should run on the full DataFrame (so 'Title' is available).
    """

    def __init__(self, title_col: str = "Title", age_col: str = "Age"):
        self.title_col = title_col
        self.age_col = age_col
        self.title_medians_ = {}
        self.global_median_ = None

    def fit(self, X, y=None):
        # Expect X as DataFrame (but coerce if necessary)
        X = pd.DataFrame(X).copy()
        # compute median age per title (ignore NaNs)
        if self.title_col in X.columns and self.age_col in X.columns:
            # convert to series to allow groupby when NaNs present
            self.title_medians_ = X.groupby(self.title_col)[self.age_col].median().to_dict()
            self.global_median_ = X[self.age_col].median()
        else:
            # fallback: no title/age info
            self.title_medians_ = {}
            self.global_median_ = np.nan
        return self

    def transform(self, X):
        """
        Vectorized transform:
         - find rows where Age is missing
         - map Title -> median
         - fill missing mapped values with global median
         - assign back to Age column
        Returns a DataFrame.
        """
        X = pd.DataFrame(X).copy()

        # if Age column not present, nothing to do
        if self.age_col not in X.columns:
            return X

        missing_mask = X[self.age_col].isna()
        if not missing_mask.any():
            return X

        # map titles to medians (NaN where title not in mapping)
        # guard against missing title column as well
        if self.title_col in X.columns:
            mapped = X.loc[missing_mask, self.title_col].map(self.title_medians_)
        else:
            # if title column missing, mapped will be all NaN
            mapped = pd.Series(index=X.loc[missing_mask].index, dtype=float)

        # replace any NaN medians with global median
        mapped = mapped.fillna(self.global_median_)

        # assign mapped ages into Age column for missing rows
        X.loc[missing_mask, self.age_col] = mapped

        return X


def make_onehot_encoder():
    """
    Return a OneHotEncoder that returns dense arrays in a version-compatible way.
    sklearn 1.2+ uses `sparse_output`, older versions use `sparse`.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# -------------------------
# ColumnTransformer builder
# -------------------------
def get_preprocessor(
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
):
    """
    Build ColumnTransformer that expects numeric features and categorical features
    to already exist in the DataFrame (i.e., after feature engineering and AgeImputer).
    """

    if numeric_features is None:
        numeric_features = ["Age", "FarePerPerson", "LogFare", "FamilySize", "TicketGroupSize"]

    if categorical_features is None:
        categorical_features = ["Sex", "Pclass", "Embarked", "Title", "Deck"]

    # Numeric pipeline: impute (median) + scale
    numeric_pipeline = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute constant + one-hot encode (dense)
    categorical_pipeline = Pipeline(
        steps=[
            ("cat_imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # ensure output is dense when possible
    )

    return preprocessor


# -------------------------
# Full pipeline (feature engineering -> age impute -> columntransformer)
# -------------------------
# Use module-level wrapper so transformer is pickleable.
feature_engineering_transformer = FunctionTransformer(_apply_create_features, validate=False)

# Build final pipeline: FE -> AgeImputer (full DF) -> ColumnTransformer (select columns)
full_pipeline = Pipeline(
    steps=[
        ("feature_engineering", feature_engineering_transformer),
        ("age_imputer_df", AgeImputer(title_col="Title", age_col="Age")),
        ("preprocessing", get_preprocessor()),
    ]
)


def get_full_pipeline():
    return full_pipeline


# -------------------------
# Quick sanity test (runs when module executed)
# -------------------------
if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_path = os.path.join(repo_root, "data", "raw", "train.csv")

    if not os.path.exists(sample_path):
        print("train.csv not found at:", sample_path)
        raise SystemExit(1)

    print("Loading", sample_path)
    df = pd.read_csv(sample_path)

    p = get_full_pipeline()
    X_trans = p.fit_transform(df)  # returns numpy array ready for model training

    print("Transformed shape:", X_trans.shape)
    print("Preprocessing test completed successfully.")