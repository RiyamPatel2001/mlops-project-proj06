"""
Online Feature Computation
ActualBudget MLOps Project — Data Pipeline
 
Takes a raw transaction (payee, amount, date) and returns a feature vector
string ready for Layer 1 inference.
 
Usage:
    python feature_computation.py --payee "WHOLE FOODS MKT" --amount 67.42 --date "2024-03-10"
 
Or as a module:
    from feature_computation import compute_features
    feature_vector = compute_features("WHOLE FOODS MKT", 67.42, "2024-03-10")
"""
 
import re
import argparse
from datetime import datetime
 
 
# ── AMOUNT BINS ───────────────────────────────────────────────────────────────
 
def bin_amount(amount: float) -> str:
    """
    Bin a transaction amount into one of five buckets.
    These bins are shared between training and inference — do not change
    without updating the training pipeline too.
    """
    if amount < 20:
        return "low"
    elif amount < 50:
        return "medium_low"
    elif amount < 150:
        return "medium"
    elif amount < 500:
        return "medium_high"
    else:
        return "high"
 
 
# ── PAYEE NORMALIZATION ───────────────────────────────────────────────────────
 
# Patterns to strip from payee strings
STRIP_PATTERNS = [
    r'#\d+',           # store numbers: #1234, #12345
    r'\*\w+',          # suffixes after asterisk: *TRIP, *ABCD123
    r'\d{4,}',         # long numeric sequences (store codes, transaction IDs)
    r'\s{2,}',         # multiple spaces → single space
]
 
def normalize_payee(payee: str) -> str:
    """
    Normalize a raw payee string for model input.
    - Uppercase
    - Strip store numbers, branch codes, transaction suffixes
    - Collapse whitespace
    """
    payee = payee.upper().strip()
    for pattern in STRIP_PATTERNS:
        payee = re.sub(pattern, ' ', payee)
    payee = payee.strip()
    return payee
 
 
# ── DAY OF WEEK ───────────────────────────────────────────────────────────────
 
def extract_day_of_week(date_str: str) -> str:
    """
    Extract day of week name from a YYYY-MM-DD date string.
    Returns e.g. 'Sunday', 'Monday', etc.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%A")
 
 
# ── MAIN FEATURE COMPUTATION ──────────────────────────────────────────────────
 
def compute_features(payee: str, amount: float, date: str) -> dict:
    """
    Compute features for a single transaction.
 
    Args:
        payee: Raw merchant name string
        amount: Transaction amount in USD
        date: Transaction date in YYYY-MM-DD format
 
    Returns:
        dict with:
            - normalized_payee: cleaned payee string
            - amount_bin: amount bucket string
            - day_of_week: day name
            - feature_vector: final model input string
    """
    normalized_payee = normalize_payee(payee)
    amount_bin = bin_amount(amount)
    day_of_week = extract_day_of_week(date)
    feature_vector = f"{normalized_payee} | amount:{amount_bin} | day:{day_of_week}"
 
    return {
        "normalized_payee": normalized_payee,
        "amount_bin": amount_bin,
        "day_of_week": day_of_week,
        "feature_vector": feature_vector,
    }
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Compute features for a transaction")
    parser.add_argument("--payee",  type=str,   required=True, help="Raw payee/merchant name")
    parser.add_argument("--amount", type=float, required=True, help="Transaction amount in USD")
    parser.add_argument("--date",   type=str,   required=True, help="Transaction date YYYY-MM-DD")
    args = parser.parse_args()
 
    result = compute_features(args.payee, args.amount, args.date)
 
    print("\n── Feature Computation Result ──────────────────")
    print(f"  Raw payee:        {args.payee}")
    print(f"  Normalized payee: {result['normalized_payee']}")
    print(f"  Amount:           ${args.amount:.2f} → {result['amount_bin']}")
    print(f"  Date:             {args.date} → {result['day_of_week']}")
    print(f"\n  Feature vector:   {result['feature_vector']}")
    print("─────────────────────────────────────────────────\n")
 
 
if __name__ == "__main__":
    main()
 
