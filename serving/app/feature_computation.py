"""
Online feature computation — ported from Saketh's data pipeline.

Takes a raw transaction (payee, amount, date) and returns a feature vector
string ready for Layer 1 inference.  The bins and normalization logic are
shared between training and inference and must stay in sync.
"""

import re
from datetime import datetime

_STRIP_PATTERNS = [
    r"#\d+",
    r"\*\w+",
    r"\d{4,}",
    r"\s{2,}",
]


def normalize_payee(payee: str) -> str:
    payee = payee.upper().strip()
    for pattern in _STRIP_PATTERNS:
        payee = re.sub(pattern, " ", payee)
    return payee.strip()


def bin_amount(amount: float) -> str:
    if amount < 20:
        return "low"
    if amount < 50:
        return "medium_low"
    if amount < 150:
        return "medium"
    if amount < 500:
        return "medium_high"
    return "high"


def day_of_week(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")


def compute_features(payee: str, amount_dollars: float, date_str: str) -> dict:
    """Build the feature vector expected by the fastText model.

    Args:
        payee: raw merchant string
        amount_dollars: signed amount in dollars (negative = expense)
        date_str: YYYY-MM-DD
    """
    norm_payee = normalize_payee(payee)
    dollars = abs(amount_dollars)
    amt_bin = bin_amount(dollars)
    dow = day_of_week(date_str)
    feature_vector = f"{norm_payee} | amount:{amt_bin} | day:{dow}"

    return {
        "normalized_payee": norm_payee,
        "amount_bin": amt_bin,
        "day_of_week": dow,
        "feature_vector": feature_vector,
    }
