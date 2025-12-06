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
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.features import create_features


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
            self.title_medians_ = X.groupby(self.title_col)[self.age_col].median().to_dict()
            self.global_median_ = X[self.age_col].median()
        else:
            # fallback: no title/age info
            self.title_medians_ = {}
            self.global_median_ = np.nan
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        def _fill_age(row):
            if pd.isna(row.get(self.age_col, np.nan)):
                title = row.get(self.title_col, None)
                med = self.title_medians_.get(title, None)
                if pd.isna(med) or med is None:
                    return self.global_median_
                return med
            return row[self.age_col]

        if self.age_col in X.columns:
            X[self.age_col] = X.apply(_fill_age, axis=1)
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
feature_engineering_transformer = FunctionTransformer(lambda X: create_features(pd.DataFrame(X)), validate=False)

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
