#!/usr/bin/env python3
"""
Batch Pipeline
ActualBudget MLOps Project — Data Pipeline

Reads the feedback store, filters for high-quality retraining candidates,
applies a temporal split, and outputs a versioned training dataset.

Retraining candidates: reviewed_by_user=TRUE AND source=layer1
These are transactions where a real user explicitly confirmed or corrected
the model's prediction — the highest quality signal for retraining.

Output: versioned CSV file ready for Riyam's training pipeline.

Usage:
    python batch_pipeline.py --feedback_url http://localhost:8000/feedback/export
    python batch_pipeline.py --feedback_file feedback_store.csv --output_dir ./datasets
    python batch_pipeline.py --feedback_file feedback_store.csv --mock

Environment variables:
    FEEDBACK_URL — URL to fetch feedback store export (optional)
"""

import os
import csv
import json
import argparse
import random
import requests
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path

random.seed(42)


# ── MOCK FEEDBACK STORE ───────────────────────────────────────────────────────

def generate_mock_feedback(n=2000):
    """
    Generates a mock feedback store for testing when the real
    feedback store (Jayraj's Postgres) is not yet available.
    """
    categories = [
        "Groceries", "Dining Out", "Transport", "Insurance", "Streaming",
        "Personal Care", "Public Transit", "Household Supplies", "Savings",
        "Entertainment", "Clothing", "Charitable Giving", "Utilities",
        "Healthcare", "Other", "Home Improvement", "Phone & Internet",
        "Tobacco", "Pets", "Rent / Mortgage", "Alcohol", "Health Insurance",
        "Travel", "Vehicle Insurance", "Property Tax", "Reading",
        "Vehicle Payment", "Childcare", "Education"
    ]
    payees = [
        "WHOLE FOODS MKT", "STARBUCKS #12345", "SHELL OIL 12345678",
        "NETFLIX.COM", "WALMART GROCERY", "MCDONALDS #54321",
        "AT&T*BILL", "UBER*TRIP", "AMAZON.COM*ABC123", "CVS PHARMACY #12345"
    ]

    rows = []
    base_date = datetime(2023, 1, 1)

    for i in range(n):
        source = "layer1" if random.random() < 0.9 else "layer2"
        reviewed = random.random() < 0.55
        correct = random.random() < 0.76

        cat = random.choice(categories)
        predicted = cat if correct else random.choice(categories)
        final = cat

        date = base_date + timedelta(days=random.randint(0, 364))

        rows.append({
            "id":                   f"fb_{i+1:06d}",
            "transaction_id":       f"txn_{i+1:07d}",
            "user_id":              f"user_{random.randint(1,499):04d}",
            "payee":                random.choice(payees),
            "amount":               -random.randint(100, 50000),
            "date":                 date.strftime("%Y-%m-%d"),
            "original_prediction":  predicted,
            "original_confidence":  round(random.uniform(0.45, 0.98), 2),
            "source":               source,
            "final_label":          final,
            "reviewed_by_user":     "True" if reviewed else "False",
            "timestamp":            (date + timedelta(hours=random.randint(1, 48))).isoformat(),
        })

    return rows


# ── LOAD FEEDBACK ─────────────────────────────────────────────────────────────

def load_feedback_from_file(filepath):
    with open(filepath) as f:
        return list(csv.DictReader(f))


def load_feedback_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    return rows


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_pipeline(feedback_rows, output_dir, version=None):
    total = len(feedback_rows)
    print(f"[batch] Loaded {total:,} feedback records")

    # Step 1: Filter retraining candidates
    # Only reviewed_by_user=TRUE AND source=layer1
    candidates = [
        r for r in feedback_rows
        if str(r.get("reviewed_by_user", "")).strip().lower() in ("true", "1", "yes")
        and r.get("source", "").strip() == "layer1"
    ]
    print(f"[batch] Retraining candidates (reviewed + layer1): {len(candidates):,} "
          f"({len(candidates)/total*100:.1f}% of feedback store)")

    if len(candidates) == 0:
        print("[batch] ERROR: No retraining candidates found. Exiting.")
        return

    # Step 2: Data quality checks
    print("[batch] Running data quality checks...")
    dropped = 0
    clean = []
    for r in candidates:
        if not r.get("final_label"):
            dropped += 1
            continue
        if not r.get("payee"):
            dropped += 1
            continue
        if not r.get("date"):
            dropped += 1
            continue
        clean.append(r)

    print(f"[batch]   Dropped {dropped} records with missing fields")
    print(f"[batch]   Clean candidates: {len(clean):,}")

    # Step 3: Class distribution check
    cat_counts = Counter(r["final_label"] for r in clean)
    print(f"[batch]   Categories represented: {len(cat_counts)}/29")
    low_cats = [(cat, count) for cat, count in cat_counts.items() if count < 5]
    if low_cats:
        print(f"[batch]   WARNING: Low sample categories: {low_cats}")

    # Step 4: Temporal split
    # Sort by date, use 80% for train, 20% for validation
    # Temporal split avoids data leakage — older data trains, newer validates
    clean.sort(key=lambda r: r["date"])
    split_idx = int(len(clean) * 0.80)
    train_rows = clean[:split_idx]
    val_rows   = clean[split_idx:]

    print(f"[batch] Temporal split:")
    print(f"[batch]   Train: {len(train_rows):,} records "
          f"({train_rows[0]['date']} to {train_rows[-1]['date']})")
    print(f"[batch]   Val:   {len(val_rows):,} records "
          f"({val_rows[0]['date']} to {val_rows[-1]['date']})")

    # Step 5: Build output — convert to training format
    # Feature vector: "PAYEE | amount:bin | day:DayName"
    def amount_bin(amount_cents):
        dollars = abs(int(amount_cents)) / 100
        if dollars < 20:    return "low"
        if dollars < 50:    return "medium_low"
        if dollars < 150:   return "medium"
        if dollars < 500:   return "medium_high"
        return "high"

    def day_of_week(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")

    def to_training_row(r, split):
        payee = str(r.get("payee", "")).strip().upper()
        amount_cents = int(r.get("amount", 0))
        date_str = r.get("date", "2023-01-01")
        feature_vector = f"{payee} | amount:{amount_bin(amount_cents)} | day:{day_of_week(date_str)}"
        return {
            "transaction_id":  r.get("transaction_id", ""),
            "user_id":         r.get("user_id", ""),
            "feature_vector":  feature_vector,
            "label":           r.get("final_label", ""),
            "split":           split,
            "source":          "feedback",
            "date":            date_str,
        }

    train_out = [to_training_row(r, "train") for r in train_rows]
    val_out   = [to_training_row(r, "val")   for r in val_rows]
    all_out   = train_out + val_out

    # Step 6: Version and save
    if version is None:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"retraining_dataset_v{version}.csv"
    manifest_file = output_dir / f"retraining_manifest_v{version}.json"

    fieldnames = ["transaction_id", "user_id", "feature_vector", "label", "split", "source", "date"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_out)

    # Save manifest
    manifest = {
        "version":          version,
        "created_at":       datetime.utcnow().isoformat(),
        "total_records":    len(all_out),
        "train_records":    len(train_out),
        "val_records":      len(val_out),
        "train_date_range": [train_rows[0]["date"], train_rows[-1]["date"]],
        "val_date_range":   [val_rows[0]["date"],   val_rows[-1]["date"]],
        "categories":       dict(cat_counts),
        "output_file":      str(output_file),
        "filter":           "reviewed_by_user=TRUE AND source=layer1",
        "split_strategy":   "temporal (80/20)",
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 50)
    print("BATCH PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Version:        v{version}")
    print(f"Output file:    {output_file}")
    print(f"Manifest:       {manifest_file}")
    print(f"Train records:  {len(train_out):,}")
    print(f"Val records:    {len(val_out):,}")
    print()
    print("Pass output_file to Riyam's training pipeline.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Pipeline — compile retraining dataset")
    parser.add_argument("--feedback_file", type=str, default=None,
                        help="Path to feedback store CSV export")
    parser.add_argument("--feedback_url",  type=str, default=None,
                        help="URL to fetch feedback store JSON export")
    parser.add_argument("--output_dir",    type=str, default="/tmp/datasets",
                        help="Directory to save versioned output")
    parser.add_argument("--version",       type=str, default=None,
                        help="Version string (default: timestamp)")
    parser.add_argument("--mock",          action="store_true",
                        help="Use mock feedback store for testing")
    args = parser.parse_args()

    print(f"[batch] Batch Pipeline starting")
    print(f"[batch] Output dir: {args.output_dir}")
    print()

    if args.mock:
        print("[batch] Using mock feedback store (2000 records)")
        feedback_rows = generate_mock_feedback(2000)
    elif args.feedback_file:
        print(f"[batch] Loading feedback from file: {args.feedback_file}")
        feedback_rows = load_feedback_from_file(args.feedback_file)
    elif args.feedback_url:
        print(f"[batch] Loading feedback from URL: {args.feedback_url}")
        feedback_rows = load_feedback_from_url(args.feedback_url)
    else:
        print("[batch] No feedback source specified. Use --mock, --feedback_file, or --feedback_url")
        return

    run_pipeline(feedback_rows, args.output_dir, args.version)


if __name__ == "__main__":
    main()
