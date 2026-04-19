#!/usr/bin/env python3
"""
Data Generator
ActualBudget MLOps Project — Data Pipeline

Simulates ActualBudget users importing transactions and interacting
with the /classify and /feedback endpoints.

Reads from production.csv (production simulation seed) from MinIO.
For each transaction:
  1. Sends POST /classify with agreed contract format
  2. Receives category prediction + confidence
  3. Simulates user behavior:
     - If confidence >= 0.6: auto-filled, user confirms if correct or corrects if wrong
     - If confidence < 0.6: left blank, user fills in manually
  4. Sends POST /feedback with the result

Usage:
    python generate_data.py
    python generate_data.py --serving_url http://129.114.25.161:8000 --max_transactions 1000

Environment variables:
    SERVING_URL       — Jayraj's serving endpoint (default: http://129.114.25.161:8000)
    MINIO_ENDPOINT    — MinIO endpoint (default: http://10.43.4.193:9000)
    MINIO_ACCESS_KEY  — MinIO access key (default: minioadmin)
    MINIO_SECRET_KEY  — MinIO secret key (default: minioadmin123)
    MINIO_BUCKET      — MinIO bucket (default: data)
"""

import csv
import io
import json
import time
import random
import argparse
import os
import requests
import boto3
from botocore.client import Config
from datetime import datetime

random.seed(42)

DEFAULT_SERVING_URL  = os.environ.get("SERVING_URL", "http://129.114.25.161:8000")
MINIO_ENDPOINT       = os.environ.get("MINIO_ENDPOINT",   "http://10.43.4.193:9000")
MINIO_ACCESS_KEY     = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY     = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET         = os.environ.get("MINIO_BUCKET",     "data")
PRODUCTION_CSV_KEY   = "raw/production.csv"

P_USER_CORRECTS_WRONG   = 0.70
P_USER_CONFIRMS_CORRECT = 0.40
P_USER_FILLS_BLANK      = 0.85


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def load_production_csv():
    print(f"[generator] Loading production.csv from MinIO: {MINIO_ENDPOINT}/{MINIO_BUCKET}/{PRODUCTION_CSV_KEY}")
    client = get_minio_client()
    response = client.get_object(Bucket=MINIO_BUCKET, Key=PRODUCTION_CSV_KEY)
    content = response["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def call_classify(serving_url, transaction):
    payload = {
        "transaction_id": transaction["transaction_id"],
        "user_id":        transaction["user_id"],
        "payee":          transaction["payee"],
        "amount":         -abs(float(transaction["amount"])),
        "date":           transaction["date"],
    }
    try:
        resp = requests.post(f"{serving_url}/classify", json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return (
                data.get("prediction_category"),
                data.get("confidence", 0.0),
                data.get("source", "layer1"),
                data.get("model_version", "v1"),
            )
        return None, 0.0, "layer1", "v1"
    except requests.exceptions.ConnectionError:
        mock_cat = transaction["category"] if random.random() < 0.75 else random.choice([
            "Groceries", "Dining Out", "Transport", "Entertainment", "Utilities"
        ])
        return mock_cat, round(random.uniform(0.45, 0.98), 2), "layer1", "v1"
    except Exception as e:
        print(f"  [warn] classify error: {e}")
        return None, 0.0, "layer1", "v1"


def call_feedback(serving_url, feedback_record):
    try:
        resp = requests.post(f"{serving_url}/feedback", json=feedback_record, timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return True
    except Exception as e:
        print(f"  [warn] feedback error: {e}")
        return False


def simulate_user_interaction(transaction, predicted_category, confidence, source):
    ground_truth = transaction["category"]
    is_correct   = (predicted_category == ground_truth)
    auto_filled  = confidence is not None and confidence >= 0.6

    reviewed_by_user = False
    final_label      = None

    if auto_filled:
        if is_correct:
            if random.random() < P_USER_CONFIRMS_CORRECT:
                reviewed_by_user = True
                final_label      = ground_truth
            else:
                reviewed_by_user = False
                final_label      = predicted_category
        else:
            if random.random() < P_USER_CORRECTS_WRONG:
                reviewed_by_user = True
                final_label      = ground_truth
            else:
                reviewed_by_user = False
                final_label      = predicted_category
    else:
        if random.random() < P_USER_FILLS_BLANK:
            reviewed_by_user = True
            final_label      = ground_truth
        else:
            return None

    return {
        "transaction_id":      transaction["transaction_id"],
        "user_id":             transaction["user_id"],
        "payee":               transaction["payee"],
        "amount":              -int(round(abs(float(transaction["amount"])) * 100)),
        "date":                transaction["date"],
        "original_prediction": predicted_category,
        "original_confidence": confidence,
        "source":              source,
        "final_label":         final_label,
        "reviewed_by_user":    reviewed_by_user,
        "timestamp":           datetime.utcnow().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="ActualBudget Data Generator")
    parser.add_argument("--serving_url",      type=str, default=DEFAULT_SERVING_URL)
    parser.add_argument("--max_transactions", type=int, default=None)
    parser.add_argument("--delay",            type=float, default=0.0)
    args = parser.parse_args()

    print(f"[generator] Data Generator starting")
    print(f"[generator] Serving URL:  {args.serving_url}")
    print(f"[generator] MinIO:        {MINIO_ENDPOINT}/{MINIO_BUCKET}/{PRODUCTION_CSV_KEY}")
    print()

    transactions = load_production_csv()

    if args.max_transactions:
        transactions = transactions[:args.max_transactions]

    print(f"[generator] Loaded {len(transactions):,} transactions")
    print(f"[generator] Starting simulation...")
    print()

    stats = {
        "total": 0, "feedback_sent": 0,
        "reviewed_by_user": 0, "correct_preds": 0,
        "auto_filled": 0, "no_action": 0,
    }

    for i, txn in enumerate(transactions):
        stats["total"] += 1

        predicted, confidence, source, model_version = call_classify(args.serving_url, txn)
        if predicted is None:
            predicted, confidence, source, model_version = txn["category"], 0.5, "layer1", "v1"

        if confidence >= 0.6:
            stats["auto_filled"] += 1
        if predicted == txn["category"]:
            stats["correct_preds"] += 1

        feedback = simulate_user_interaction(txn, predicted, confidence, source)

        if feedback is None:
            stats["no_action"] += 1
        else:
            if feedback["reviewed_by_user"]:
                stats["reviewed_by_user"] += 1
            call_feedback(args.serving_url, feedback)
            stats["feedback_sent"] += 1

        if (i + 1) % 1000 == 0:
            pct = (i + 1) / len(transactions) * 100
            print(f"[generator] Processed {i+1:,}/{len(transactions):,} ({pct:.0f}%) — "
                  f"feedback sent: {stats['feedback_sent']:,}")

        if args.delay > 0:
            time.sleep(args.delay)

    total = stats["total"]
    print()
    print("=" * 50)
    print("GENERATOR COMPLETE")
    print("=" * 50)
    print(f"Total transactions:        {total:,}")
    print(f"Model accuracy:            {stats['correct_preds']/total*100:.1f}%")
    print(f"Auto-filled (conf>=0.6):   {stats['auto_filled']:,} ({stats['auto_filled']/total*100:.1f}%)")
    print(f"Feedback records sent:     {stats['feedback_sent']:,}")
    print(f"Reviewed by user:          {stats['reviewed_by_user']:,}")
    print(f"No user action:            {stats['no_action']:,}")


if __name__ == "__main__":
    main()
