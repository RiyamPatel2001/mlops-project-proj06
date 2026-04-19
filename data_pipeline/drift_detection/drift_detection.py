#!/usr/bin/env python3
"""
Drift Detection
ActualBudget MLOps Project — Data Pipeline

Monitors live inference data quality and drift in production.
Compares incoming transaction distributions from the feedback store
against the training data baseline (train.csv from MinIO).

Drift is detected on:
  1. Category distribution — are users seeing different category predictions?
  2. Amount bin distribution — are transaction amounts shifting?
  3. Payee vocabulary drift — are new payees appearing not seen in training?

Output: drift report JSON uploaded to MinIO at data/drift/drift_report_v{timestamp}.json

Usage:
    python drift_detection.py
    python drift_detection.py --mock

Environment variables:
    FEEDBACK_URL      — URL to fetch live inference data
    MINIO_ENDPOINT    — MinIO endpoint (default: http://10.43.4.193:9000)
    MINIO_ACCESS_KEY  — MinIO access key (default: minioadmin)
    MINIO_SECRET_KEY  — MinIO secret key (default: minioadmin123)
    MINIO_BUCKET      — MinIO bucket (default: data)
"""

import os
import io
import csv
import json
import random
import argparse
import requests
import boto3
from botocore.client import Config
from datetime import datetime
from collections import Counter
from math import log, sqrt

random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────

FEEDBACK_URL     = os.environ.get("FEEDBACK_URL",     None)
MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://10.43.4.193:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "data")
BASELINE_KEY     = "raw/train.csv"
DRIFT_PREFIX     = "drift"

# Drift thresholds
CATEGORY_DRIFT_THRESHOLD = 0.15   # 15% JS divergence triggers warning
AMOUNT_DRIFT_THRESHOLD   = 0.15
PAYEE_NEW_RATIO_THRESHOLD = 0.30  # 30% new payees triggers warning


# ── MINIO ─────────────────────────────────────────────────────────────────────

def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def load_csv_from_minio(client, key):
    response = client.get_object(Bucket=MINIO_BUCKET, Key=key)
    content = response["Body"].read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(content)))


def upload_json_to_minio(client, content, key):
    client.put_object(
        Bucket=MINIO_BUCKET,
        Key=key,
        Body=json.dumps(content, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[drift] Uploaded report to MinIO: s3://{MINIO_BUCKET}/{key}")


# ── MOCK DATA ─────────────────────────────────────────────────────────────────

def generate_mock_live_data(n=500):
    categories = [
        "Groceries", "Dining Out", "Transport", "Streaming",
        "Healthcare", "Utilities", "Phone & Internet", "Entertainment",
        "Clothing", "Travel", "Savings", "Other"
    ]
    payees = [
        "WHOLE FOODS MKT", "NETFLIX.COM", "UBER*TRIP", "STARBUCKS",
        "CVS PHARMACY", "AT&T*BILL", "AMAZON.COM", "WALMART",
        "NEW PAYEE 001", "NEW PAYEE 002", "NEW PAYEE 003"
    ]
    rows = []
    base_date = datetime(2024, 1, 1)
    for i in range(n):
        amount = -random.randint(100, 50000)
        rows.append({
            "transaction_id":      f"txn_{i+1:07d}",
            "user_id":             f"user_{random.randint(1,499):04d}",
            "payee":               random.choice(payees),
            "amount":              amount,
            "date":                (base_date).strftime("%Y-%m-%d"),
            "original_prediction": random.choice(categories),
            "original_confidence": round(random.uniform(0.45, 0.98), 2),
            "source":              "layer1",
            "final_label":         random.choice(categories),
            "reviewed_by_user":    "True",
            "timestamp":           datetime.utcnow().isoformat(),
        })
    return rows


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def amount_bin(amount_raw):
    """Bin amount in cents to dollar buckets."""
    dollars = abs(float(amount_raw))
    if dollars < 20:   return "low"
    if dollars < 50:   return "medium_low"
    if dollars < 150:  return "medium"
    if dollars < 500:  return "medium_high"
    return "high"


def get_category_distribution(rows, field="category"):
    counts = Counter(r.get(field, "Unknown") for r in rows)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}


def get_amount_distribution(rows, field="amount"):
    counts = Counter(amount_bin(r.get(field, 0)) for r in rows)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}


def get_payee_vocabulary(rows, field="payee"):
    return set(str(r.get(field, "")).strip().upper() for r in rows if r.get(field))


# ── DRIFT METRICS ─────────────────────────────────────────────────────────────

def js_divergence(p_dist, q_dist):
    """
    Jensen-Shannon divergence between two distributions.
    Returns a value between 0 (identical) and 1 (completely different).
    JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
    """
    all_keys = set(p_dist.keys()) | set(q_dist.keys())
    eps = 1e-10

    p = {k: p_dist.get(k, 0) for k in all_keys}
    q = {k: q_dist.get(k, 0) for k in all_keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in all_keys}

    def kl(a, b):
        return sum(
            a[k] * log((a[k] + eps) / (b[k] + eps))
            for k in all_keys if a[k] > 0
        )

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return round(min(jsd, 1.0), 4)


def payee_drift(baseline_vocab, live_vocab):
    """Fraction of live payees not seen in training."""
    if not live_vocab:
        return 0.0
    new_payees = live_vocab - baseline_vocab
    return round(len(new_payees) / len(live_vocab), 4)


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_drift_detection(baseline_rows, live_rows):
    print(f"[drift] Baseline records:   {len(baseline_rows):,}")
    print(f"[drift] Live records:       {len(live_rows):,}")
    print()

    alerts = []
    warnings = []

    # 1. Category distribution drift
    print("[drift] Checking category distribution drift...")
    baseline_cat = get_category_distribution(baseline_rows, field="category")
    live_cat = get_category_distribution(live_rows, field="original_prediction")
    cat_jsd = js_divergence(baseline_cat, live_cat)
    print(f"[drift]   Category JSD: {cat_jsd:.4f} (threshold: {CATEGORY_DRIFT_THRESHOLD})")

    if cat_jsd > CATEGORY_DRIFT_THRESHOLD:
        msg = f"ALERT: Category distribution drift detected — JSD={cat_jsd:.4f} exceeds threshold {CATEGORY_DRIFT_THRESHOLD}"
        print(f"[drift]   {msg}")
        alerts.append(msg)
    else:
        print(f"[drift]   Category distribution OK")

    # 2. Amount bin distribution drift
    print("[drift] Checking amount distribution drift...")
    baseline_amt = get_amount_distribution(baseline_rows, field="amount")
    live_amt = get_amount_distribution(live_rows, field="amount")
    amt_jsd = js_divergence(baseline_amt, live_amt)
    print(f"[drift]   Amount JSD: {amt_jsd:.4f} (threshold: {AMOUNT_DRIFT_THRESHOLD})")

    if amt_jsd > AMOUNT_DRIFT_THRESHOLD:
        msg = f"ALERT: Amount distribution drift detected — JSD={amt_jsd:.4f} exceeds threshold {AMOUNT_DRIFT_THRESHOLD}"
        print(f"[drift]   {msg}")
        alerts.append(msg)
    else:
        print(f"[drift]   Amount distribution OK")

    # 3. Payee vocabulary drift
    print("[drift] Checking payee vocabulary drift...")
    baseline_vocab = get_payee_vocabulary(baseline_rows, field="payee")
    live_vocab = get_payee_vocabulary(live_rows, field="payee")
    new_ratio = payee_drift(baseline_vocab, live_vocab)
    new_payees = live_vocab - baseline_vocab
    print(f"[drift]   New payees: {len(new_payees)} / {len(live_vocab)} ({new_ratio*100:.1f}%)")
    print(f"[drift]   New payee ratio: {new_ratio:.4f} (threshold: {PAYEE_NEW_RATIO_THRESHOLD})")

    if new_ratio > PAYEE_NEW_RATIO_THRESHOLD:
        msg = f"WARNING: High new payee ratio — {new_ratio*100:.1f}% of live payees unseen in training"
        print(f"[drift]   {msg}")
        warnings.append(msg)
    else:
        print(f"[drift]   Payee vocabulary OK")

    # 4. Data quality checks on live data
    print("[drift] Checking live data quality...")
    missing_payee = sum(1 for r in live_rows if not r.get("payee"))
    missing_amount = sum(1 for r in live_rows if not r.get("amount"))
    low_confidence = sum(
        1 for r in live_rows
        if r.get("original_confidence") and float(r.get("original_confidence", 0)) < 0.3
    )
    print(f"[drift]   Missing payee:   {missing_payee}")
    print(f"[drift]   Missing amount:  {missing_amount}")
    print(f"[drift]   Low confidence (<0.3): {low_confidence} ({low_confidence/len(live_rows)*100:.1f}%)")

    if low_confidence / len(live_rows) > 0.20:
        msg = f"WARNING: {low_confidence/len(live_rows)*100:.1f}% of live predictions have confidence < 0.3"
        warnings.append(msg)

    # Build report
    report = {
        "version":              datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "created_at":           datetime.utcnow().isoformat(),
        "baseline_records":     len(baseline_rows),
        "live_records":         len(live_rows),
        "drift_status":         "ALERT" if alerts else ("WARNING" if warnings else "OK"),
        "alerts":               alerts,
        "warnings":             warnings,
        "metrics": {
            "category_jsd":         cat_jsd,
            "amount_jsd":           amt_jsd,
            "new_payee_ratio":      new_ratio,
            "new_payee_count":      len(new_payees),
            "live_vocab_size":      len(live_vocab),
            "baseline_vocab_size":  len(baseline_vocab),
            "missing_payee":        missing_payee,
            "missing_amount":       missing_amount,
            "low_confidence_count": low_confidence,
        },
        "thresholds": {
            "category_jsd":      CATEGORY_DRIFT_THRESHOLD,
            "amount_jsd":        AMOUNT_DRIFT_THRESHOLD,
            "new_payee_ratio":   PAYEE_NEW_RATIO_THRESHOLD,
        },
        "baseline_category_dist": baseline_cat,
        "live_category_dist":     live_cat,
        "baseline_amount_dist":   baseline_amt,
        "live_amount_dist":       live_amt,
    }

    return report


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drift Detection — monitor live inference data")
    parser.add_argument("--feedback_url", type=str,
                        default=os.environ.get("FEEDBACK_URL", None),
                        help="URL to fetch live feedback store export")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock live data for testing")
    args = parser.parse_args()

    print(f"[drift] Drift Detection starting")
    print(f"[drift] MinIO endpoint: {MINIO_ENDPOINT}")
    print(f"[drift] Baseline:       s3://{MINIO_BUCKET}/{BASELINE_KEY}")
    print()

    minio = get_minio_client()

    # Load baseline
    print(f"[drift] Loading baseline from MinIO...")
    baseline_rows = load_csv_from_minio(minio, BASELINE_KEY)
    print(f"[drift] Baseline loaded: {len(baseline_rows):,} records")

    # Load live data
    if args.mock:
        print(f"[drift] Using mock live data (500 records)")
        live_rows = generate_mock_live_data(500)
    elif args.feedback_url:
        print(f"[drift] Loading live data from: {args.feedback_url}")
        resp = requests.get(args.feedback_url, timeout=10)
        resp.raise_for_status()
        live_rows = resp.json()
        print(f"[drift] Live data loaded: {len(live_rows):,} records")
    else:
        print("[drift] No live data source specified. Use --mock or --feedback_url")
        return

    if len(live_rows) < 10:
        print(f"[drift] ERROR: Only {len(live_rows)} live records — not enough for drift detection (minimum 10)")
        return

    print()

    # Run drift detection
    report = run_drift_detection(baseline_rows, live_rows)

    # Upload report to MinIO
    version = report["version"]
    report_key = f"{DRIFT_PREFIX}/drift_report_v{version}.json"
    upload_json_to_minio(minio, report, report_key)

    print()
    print("=" * 50)
    print("DRIFT DETECTION COMPLETE")
    print("=" * 50)
    print(f"Status:         {report['drift_status']}")
    print(f"Category JSD:   {report['metrics']['category_jsd']}")
    print(f"Amount JSD:     {report['metrics']['amount_jsd']}")
    print(f"New payees:     {report['metrics']['new_payee_ratio']*100:.1f}%")
    if report["alerts"]:
        print()
        print("ALERTS:")
        for a in report["alerts"]:
            print(f"  {a}")
    if report["warnings"]:
        print()
        print("WARNINGS:")
        for w in report["warnings"]:
            print(f"  {w}")
    print()
    print(f"Full report:    s3://{MINIO_BUCKET}/{report_key}")


if __name__ == "__main__":
    main()
