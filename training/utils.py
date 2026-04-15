"""
utils.py
--------
Shared preprocessing utilities for the Layer-1 training pipeline.

normalize_payee and user_stratified_split are kept here so train.py
can import them without depending on the data pipeline directory.
"""

import re

import numpy as np
import pandas as pd


# ── Payee normalization ───────────────────────────────────────────────────────

def normalize_payee(s: str) -> str:
    """
    Collapse merchant variants to a canonical form.
    Steps (order matters):
      1. Uppercase + strip
      2. Remove store/branch numbers  (#7548, # 03, etc.)
      3. Remove platform order suffixes  (*ABCDEF (Uber, Amazon, etc.))
      4. Remove trailing digits left after step 2
      5. Collapse whitespace
    """
    s = s.upper().strip()
    s = re.sub(r'#\s*\d+', '', s)
    s = re.sub(r'\*\S+', '', s)
    s = re.sub(r'\s\d+$', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── User-stratified split (Strategy D) ───────────────────────────────────────

def user_stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by user_id so no user appears in both train and val.

    Rationale: simulates the new-user generalisation scenario — the model
    must categorise transactions for people it has never seen.  This is the
    honest eval for a shared Layer-1 model (Strategy D from EDA).

    Returns:
        df_train, df_val
    """
    rng = np.random.default_rng(random_state)
    all_users = df["user_id"].unique()
    n_val = int(len(all_users) * val_frac)
    val_users = set(rng.choice(all_users, size=n_val, replace=False))

    df_train = df[~df["user_id"].isin(val_users)].reset_index(drop=True)
    df_val   = df[ df["user_id"].isin(val_users)].reset_index(drop=True)
    return df_train, df_val
