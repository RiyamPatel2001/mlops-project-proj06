#!/usr/bin/env python3
"""
Batch Pipeline
ActualBudget MLOps Project — Data Pipeline

Reads the feedback store, filters for high-quality retraining candidates,
applies a temporal split, and outputs a versioned training dataset to MinIO.

Retraining candidates: reviewed_by_user=TRUE AND source=layer1
These are transactions where a real user explicitly confirmed or corrected
the model's prediction — the highest quality signal for retraining.

Output: versioned CSV file uploaded to MinIO at data/retraining/

Usage:
    python batch_pipeline.py --feedback_url http://JAYRAJ_URL:8000/feedback/export
    python batch_pipeline.py --feedback_file feedback_store.csv
    python batch_pipeline.py --mock

Environment variables:
    FEEDBACK_URL      — URL to fetch feedback store export
    MINIO_ENDPOINT    — MinIO endpoint (default: http://10.43.4.193:9000)
    MINIO_ACCESS_KEY  — MinIO access key (default: minioadmin)
    MINIO_SECRET_KEY  — MinIO secret key (default: minioadmin123)
    MINIO_BUCKET      — MinIO bucket (default: data)
"""

import os
import csv
import json
import argparse
import random
import io
import requests
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path

import boto3
from botocore.client import Config

random.seed(42)

# ── MINIO CONFIG ──────────────────────────────────────────────────────────────

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://10.43.4.193:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "data")
MINIO_PREFIX     = "retraining"


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_to_minio(client, content: str, key: str):
    client.put_object(
        Bucket=MINIO_BUCKET,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType="text/csv",
    )
    print(f"[batch] Uploaded to MinIO: s3://{MINIO_BUCKET}/{key}")


def upload_json_to_minio(client, content: dict, key: str):
    client.put_object(
        Bucket=MINIO_BUCKET,
        Key=key,
        Body=json.dumps(content, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[batch] Uploaded to MinIO: s3://{MINIO_BUCKET}/{key}")


# ── MOCK FEEDBACK STORE ───────────────────────────────────────────────────────

def generate_mock_feedback(n=2000):
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
        source   = "layer1" if random.random() < 0.9 else "layer2"
        reviewed = random.random() < 0.55
        correct  = random.random() < 0.76

        cat       = random.choice(categories)
        predicted = cat if correct else random.choice(categories)
        final     = cat
        date      = base_date + timedelta(days=random.randint(0, 364))

        rows.append({
            "id":                  f"fb_{i+1:06d}",
            "transaction_id":      f"txn_{i+1:07d}",
            "user_id":             f"user_{random.randint(1,499):04d}",
            "payee":               random.choice(payees),
            "amount":              -random.randint(100, 50000),
            "date":                date.strftime("%Y-%m-%d"),
            "original_prediction": predicted,
            "original_confidence": round(random.uniform(0.45, 0.98), 2),
            "source":              source,
            "final_label":         final,
            "reviewed_by_user":    "True" if reviewed else "False",
            "timestamp":           (date + timedelta(hours=random.randint(1, 48))).isoformat(),
        })

    return rows


# ── LOAD FEEDBACK ─────────────────────────────────────────────────────────────

def load_feedback_from_file(filepath):
    with open(filepath) as f:
        return list(csv.DictReader(f))


def load_feedback_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_pipeline(feedback_rows, version=None):
    total = len(feedback_rows)
    print(f"[batch] Loaded {total:,} feedback records")

    # Step 1: Filter retraining candidates
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

    # Step 2: Data quality gates
    print("[batch] Running data quality gates...")
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

    # Minimum records gate
    if len(clean) < 100:
        print(f"[batch] ERROR: Only {len(clean)} clean records — minimum is 100. Exiting.")
        return

    # Step 3: Class distribution check
    cat_counts = Counter(r["final_label"] for r in clean)
    print(f"[batch]   Categories represented: {len(cat_counts)}/29")
    low_cats = [(cat, count) for cat, count in cat_counts.items() if count < 5]
    if low_cats:
        print(f"[batch]   WARNING: Low sample categories: {low_cats}")

    # Category coverage gate
    if len(cat_counts) < 10:
        print(f"[batch] ERROR: Only {len(cat_counts)} categories represented — minimum is 10. Exiting.")
        return

    # Step 4: Temporal split
    clean.sort(key=lambda r: r["date"])
    split_idx  = int(len(clean) * 0.80)
    train_rows = clean[:split_idx]
    val_rows   = clean[split_idx:]

    print(f"[batch] Temporal split:")
    print(f"[batch]   Train: {len(train_rows):,} records "
          f"({train_rows[0]['date']} to {train_rows[-1]['date']})")
    print(f"[batch]   Val:   {len(val_rows):,} records "
          f"({val_rows[0]['date']} to {val_rows[-1]['date']})")

    # Step 5: Build training format
    def amount_bin(amount_cents):
        dollars = abs(int(amount_cents)) / 100
        if dollars < 20:   return "low"
        if dollars < 50:   return "medium_low"
        if dollars < 150:  return "medium"
        if dollars < 500:  return "medium_high"
        return "high"

    def day_of_week(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")

    def to_training_row(r, split):
        payee          = str(r.get("payee", "")).strip().upper()
        amount_cents   = int(r.get("amount", 0))
        date_str       = r.get("date", "2023-01-01")
        feature_vector = f"{payee} | amount:{amount_bin(amount_cents)} | day:{day_of_week(date_str)}"
        return {
            "transaction_id": r.get("transaction_id", ""),
            "user_id":        r.get("user_id", ""),
            "feature_vector": feature_vector,
            "label":          r.get("final_label", ""),
            "split":          split,
            "source":         "feedback",
            "date":           date_str,
        }

    train_out = [to_training_row(r, "train") for r in train_rows]
    val_out   = [to_training_row(r, "val")   for r in val_rows]
    all_out   = train_out + val_out

    # Step 6: Version and upload to MinIO
    if version is None:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    csv_key      = f"{MINIO_PREFIX}/retraining_dataset_v{version}.csv"
    manifest_key = f"{MINIO_PREFIX}/retraining_manifest_v{version}.json"

    # Build CSV string
    fieldnames = ["transaction_id", "user_id", "feature_vector", "label", "split", "source", "date"]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_out)
    csv_content = csv_buffer.getvalue()

    # Build manifest
    manifest = {
        "version":          version,
        "created_at":       datetime.utcnow().isoformat(),
        "total_records":    len(all_out),
        "train_records":    len(train_out),
        "val_records":      len(val_out),
        "train_date_range": [train_rows[0]["date"], train_rows[-1]["date"]],
        "val_date_range":   [val_rows[0]["date"],   val_rows[-1]["date"]],
        "categories":       dict(cat_counts),
        "minio_path":       f"s3://{MINIO_BUCKET}/{csv_key}",
        "filter":           "reviewed_by_user=TRUE AND source=layer1",
        "split_strategy":   "temporal (80/20)",
    }

    # Upload both to MinIO
    minio = get_minio_client()
    upload_to_minio(minio, csv_content, csv_key)
    upload_json_to_minio(minio, manifest, manifest_key)

    print()
    print("=" * 50)
    print("BATCH PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Version:        v{version}")
    print(f"MinIO path:     s3://{MINIO_BUCKET}/{csv_key}")
    print(f"Train records:  {len(train_out):,}")
    print(f"Val records:    {len(val_out):,}")
    print()
    print("Riyam's training job reads from:")
    print(f"  s3://{MINIO_BUCKET}/{csv_key}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Pipeline — compile retraining dataset")
    parser.add_argument("--feedback_file", type=str, default=None,
                        help="Path to feedback store CSV export")
    parser.add_argument("--feedback_url",  type=str,
                        default=os.environ.get("FEEDBACK_URL", None),
                        help="URL to fetch feedback store JSON export")
    parser.add_argument("--version",       type=str, default=None,
                        help="Version string (default: timestamp)")
    parser.add_argument("--mock",          action="store_true",
                        help="Use mock feedback store for testing")
    args = parser.parse_args()

    print(f"[batch] Batch Pipeline starting")
    print(f"[batch] MinIO endpoint: {MINIO_ENDPOINT}")
    print(f"[batch] Output bucket:  {MINIO_BUCKET}/{MINIO_PREFIX}/")
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

    run_pipeline(feedback_rows, args.version)


if __name__ == "__main__":
    main()
