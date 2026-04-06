#!/usr/bin/env python3
"""
CEX Ingestion Pipeline
ActualBudget MLOps Project — Data Pipeline

Downloads CEX PUMD data from BLS, generates synthetic transactions,
and uploads to Chameleon object storage.

Usage:
    python ingest.py --year 2022
    python ingest.py --year 2023
    python ingest.py --year 2024

Environment variables required:
    APP_CRED_ID      — Chameleon application credential ID
    APP_CRED_SECRET  — Chameleon application credential secret
    OS_BUCKET        — Object storage bucket name (default: sr7714-data-proj06)
"""

import os
import sys
import zipfile
import argparse
import subprocess
from pathlib import Path

OS_AUTH_URL     = os.getenv("OS_AUTH_URL", "https://chi.tacc.chameleoncloud.org:5000/v3")
OS_BUCKET       = os.getenv("OS_BUCKET", "sr7714-data-proj06")
APP_CRED_ID     = os.getenv("APP_CRED_ID")
APP_CRED_SECRET = os.getenv("APP_CRED_SECRET")

CEX_URLS = {
    2022: {
        "interview": "https://www.bls.gov/cex/pumd/data/csv/intrvw22.zip",
        "diary":     "https://www.bls.gov/cex/pumd/data/csv/diary22.zip",
    },
    2023: {
        "interview": "https://www.bls.gov/cex/pumd/data/csv/intrvw23.zip",
        "diary":     "https://www.bls.gov/cex/pumd/data/csv/diary23.zip",
    },
    2024: {
        "interview": "https://www.bls.gov/cex/pumd/data/csv/intrvw24.zip",
        "diary":     "https://www.bls.gov/cex/pumd/data/csv/diary24.zip",
    },
}

CEX_FILES = {
    2022: {
        "interview": ["fmli222.csv", "fmli223.csv", "fmli224.csv", "fmli231.csv"],
        "diary":     ["fmld221.csv", "fmld222.csv", "fmld223.csv", "fmld224.csv"],
    },
    2023: {
        "interview": ["fmli232.csv", "fmli233.csv", "fmli234.csv", "fmli241.csv"],
        "diary":     ["fmld231.csv", "fmld232.csv", "fmld233.csv", "fmld234.csv"],
    },
    2024: {
        "interview": ["fmli241x.csv", "fmli242.csv", "fmli243.csv", "fmli244.csv", "fmli251.csv"],
        "diary":     ["fmld241.csv", "fmld242.csv", "fmld243.csv", "fmld244.csv"],
    },
}

OUTPUT_NAMES = {
    2022: "transactions_2022.csv",
    2023: "transactions_2023.csv",
    2024: "transactions_2024.csv",
}


def log(msg):
    print(f"[ingest] {msg}", flush=True)


def download_file(url, dest):
    log(f"Downloading {url}")
    cmd = [
        "curl", "-L", "-o", str(dest),
        "--user-agent", "Mozilla/5.0 (compatible; MLOps-proj06/1.0)",
        "--retry", "3",
        "--retry-delay", "2",
        "--progress-bar",
        url
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"curl download failed for {url}")
    log(f"Saved to {dest}")


def extract_files(zip_path, target_files, extract_dir):
    log(f"Extracting from {zip_path}")
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as z:
        all_names = z.namelist()
        for target in target_files:
            matches = [n for n in all_names if n.endswith(target) or n.lower().endswith(target.lower())]
            if not matches:
                log(f"  WARNING: {target} not found in zip")
                continue
            match = matches[0]
            z.extract(match, extract_dir)
            extracted_path = Path(extract_dir) / match
            flat_path = Path(extract_dir) / target
            if str(extracted_path) != str(flat_path):
                flat_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_path.rename(flat_path)
            extracted.append(str(flat_path))
            log(f"  Extracted {target}")
    return extracted


def upload_to_swift(local_path, object_name):
    log(f"Uploading to {OS_BUCKET}/{object_name}")
    cmd = [
        "swift",
        "--os-auth-url", OS_AUTH_URL,
        "--os-auth-type", "v3applicationcredential",
        "--os-application-credential-id", APP_CRED_ID,
        "--os-application-credential-secret", APP_CRED_SECRET,
        "upload", OS_BUCKET, local_path,
        "--object-name", object_name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Swift upload failed: {result.stderr}")
    log(f"Uploaded successfully")


def list_bucket():
    cmd = [
        "swift",
        "--os-auth-url", OS_AUTH_URL,
        "--os-auth-type", "v3applicationcredential",
        "--os-application-credential-id", APP_CRED_ID,
        "--os-application-credential-secret", APP_CRED_SECRET,
        "list", OS_BUCKET,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        files = [f for f in result.stdout.strip().split("\n") if f]
        log(f"Bucket {OS_BUCKET} contents ({len(files)} files):")
        for f in files:
            log(f"  {f}")
    else:
        log(f"Could not list bucket: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="CEX Ingestion Pipeline")
    parser.add_argument("--year",    type=int, required=True, choices=[2022, 2023, 2024])
    parser.add_argument("--n_users", type=int, default=500)
    parser.add_argument("--workdir", type=str, default="/tmp/cex_ingest")
    args = parser.parse_args()

    if not APP_CRED_ID or not APP_CRED_SECRET:
        log("ERROR: APP_CRED_ID and APP_CRED_SECRET must be set as environment variables")
        sys.exit(1)

    year    = args.year
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    log(f"Starting CEX ingestion pipeline for {year}")
    log(f"Work directory: {workdir}")
    log(f"Target bucket:  {OS_BUCKET}")

    # Step 1: Download
    log("=" * 50)
    log("STEP 1: Downloading CEX data from BLS")
    interview_zip = workdir / f"intrvw{str(year)[2:]}.zip"
    diary_zip     = workdir / f"diary{str(year)[2:]}.zip"
    download_file(CEX_URLS[year]["interview"], interview_zip)
    download_file(CEX_URLS[year]["diary"],     diary_zip)

    # Step 2: Extract
    log("=" * 50)
    log("STEP 2: Extracting FMLI and FMLD files")
    extract_dir = workdir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    interview_files = extract_files(interview_zip, CEX_FILES[year]["interview"], extract_dir)
    diary_files     = extract_files(diary_zip,     CEX_FILES[year]["diary"],     extract_dir)

    if not interview_files or not diary_files:
        log("ERROR: Failed to extract required files")
        sys.exit(1)

    # Step 3: Generate transactions
    log("=" * 50)
    log("STEP 3: Generating synthetic transactions")
    output_csv  = workdir / OUTPUT_NAMES[year]
    script_path = Path(__file__).parent / "generate_transactions.py"

    cmd = (
        [sys.executable, str(script_path),
         "--year", str(year),
         "--interview_files"] + interview_files +
        ["--diary_files"] + diary_files +
        ["--output", str(output_csv),
         "--n_users", str(args.n_users)]
    )

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log("ERROR: generate_transactions.py failed")
        sys.exit(1)

    if not output_csv.exists():
        log(f"ERROR: Output file not found: {output_csv}")
        sys.exit(1)

    size_mb = output_csv.stat().st_size / 1024 / 1024
    log(f"Generated {OUTPUT_NAMES[year]} ({size_mb:.1f} MB)")

    # Step 4: Upload
    log("=" * 50)
    log("STEP 4: Uploading to Chameleon object storage")
    upload_to_swift(str(output_csv), OUTPUT_NAMES[year])

    # Step 5: Confirm
    log("=" * 50)
    log("STEP 5: Confirming upload")
    list_bucket()

    log("=" * 50)
    log(f"PIPELINE COMPLETE — {OUTPUT_NAMES[year]} is live in {OS_BUCKET}")


if __name__ == "__main__":
    main()
