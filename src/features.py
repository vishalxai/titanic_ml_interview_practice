# src/features.py
"""
Feature engineering utilities for Titanic ML project.

Functions
---------
- extract_title(name) -> str
- normalize_title(title) -> str
- extract_deck(cabin) -> str
- compute_ticket_group_size(df, ticket_col='Ticket') -> pd.Series
- create_features(df) -> pd.DataFrame

Usage:
    from src.features import create_features
    df = pd.read_csv("data/raw/train.csv")
    df_fe = create_features(df)
"""

from typing import Optional
import re

import numpy as np
import pandas as pd


def extract_title(name: Optional[str]) -> str:
    """Extract title from a passenger Name string.
    Examples: 'Braund, Mr. Owen Harris' -> 'Mr'
    Returns 'Unknown' if title cannot be found.
    """
    if pd.isna(name):
        return "Unknown"
    # Look for a title like 'Mr.', 'Mrs.', 'Miss.', 'Master.', etc.
    m = re.search(r",\s*([^\.]+)\.", name)
    if m:
        return m.group(1).strip()
    return "Unknown"


def normalize_title(title: str) -> str:
    """Map rare/variant titles to a small set of common titles."""
    t = title.lower()
    if t in ("mr",):
        return "Mr"
    if t in ("mrs", "misses"):
        return "Mrs"
    if t in ("miss",):
        return "Miss"
    if t in ("master",):
        return "Master"
    # common variants that map to 'Rare'
    rare = {
        "dr", "rev", "col", "major", "lady", "sir", "mlle", "mme", "capt",
        "countess", "jonkheer", "don", "dona"
    }
    if t in rare:
        return "Rare"
    # fallback: capitalize first letter
    return title.title()


def extract_deck(cabin: Optional[str]) -> str:
    """Extract deck letter from Cabin (first character). If missing, return 'Unknown'."""
    if pd.isna(cabin) or (not isinstance(cabin, str)) or cabin.strip() == "":
        return "Unknown"
    # Cabin may contain values like 'C85' or 'C85 C123' - first letter is deck
    return cabin.strip()[0].upper()


def compute_ticket_group_size(df: pd.DataFrame, ticket_col: str = "Ticket") -> pd.Series:
    """Return a Series with the count of passengers sharing the same ticket."""
    if ticket_col not in df.columns:
        raise KeyError(f"{ticket_col} not found in DataFrame")
    return df.groupby(ticket_col)[ticket_col].transform("count").astype(int)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features on a copy of the input DataFrame.

    Adds columns:
    - Title (normalized)
    - FamilySize (SibSp + Parch + 1)
    - IsAlone (1 if FamilySize==1 else 0)
    - TicketGroupSize (count of same Ticket)
    - FarePerPerson (Fare / FamilySize; NaN preserved if Fare missing)
    - LogFare (np.log1p(Fare))
    - Deck (first letter of Cabin or 'Unknown')

    Notes:
    - This function does NOT impute Age or Fare; imputation should be handled
      later inside the preprocessing pipeline (so transformations remain deterministic).
    - The function returns a new DataFrame (does not modify in-place).
    """
    df = df.copy()

    # Title
    df["Title"] = df["Name"].apply(extract_title)
    df["Title"] = df["Title"].apply(normalize_title)

    # Family size & alone
    df["FamilySize"] = df["SibSp"].fillna(0).astype(int) + df["Parch"].fillna(0).astype(int) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Ticket group size
    df["TicketGroupSize"] = compute_ticket_group_size(df, ticket_col="Ticket")

    # Fare per person
    # Keep NaN if Fare is missing; we will impute later in pipeline.
    df["FarePerPerson"] = np.where(df["FamilySize"] > 0, df["Fare"] / df["FamilySize"], df["Fare"])
    # Log Fare (handle NaN automatically)
    df["LogFare"] = np.log1p(df["Fare"].fillna(0))

    # Deck from Cabin
    df["Deck"] = df["Cabin"].apply(extract_deck)

    return df


# Quick test / CLI
if __name__ == "__main__":
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_path = os.path.join(repo_root, "data", "raw", "train.csv")

    if not os.path.exists(sample_path):
        print(f"Train file not found at: {sample_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading", sample_path)
    df = pd.read_csv(sample_path)
    df_fe = create_features(df)

    # show a small summary
    cols_to_show = ["PassengerId", "Name", "Title", "FamilySize", "IsAlone", "Ticket", "TicketGroupSize", "Fare", "FarePerPerson", "LogFare", "Cabin", "Deck"]
    print(df_fe[cols_to_show].head(8).to_string(index=False))