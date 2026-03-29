"""
Compile individual model evaluation CSVs into a single serving options table.

Reads per-model results from results/<model>_evaluation.csv and the FastAPI
benchmark from results/fastapi_benchmark.csv, then produces a unified
results/serving_options_table_final.csv.

Usage:
    python compile_results.py
"""

import os
import pandas as pd
import glob


def main():
    results_dir = "results"
    all_dfs = []

    eval_csvs = glob.glob(os.path.join(results_dir, "*_evaluation.csv"))
    for csv_path in sorted(eval_csvs):
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
        print(f"Loaded {csv_path}: {len(df)} rows")

    if not all_dfs:
        print("No evaluation CSVs found. Run the evaluation notebooks first.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    fastapi_csv = os.path.join(results_dir, "fastapi_benchmark.csv")
    if os.path.exists(fastapi_csv):
        fastapi_df = pd.read_csv(fastapi_csv)
        print(f"Loaded {fastapi_csv}: {len(fastapi_df)} rows")
    else:
        print(f"FastAPI benchmark not found at {fastapi_csv}")

    combined = combined.sort_values(
        ["model", "p50_latency_ms"],
        ascending=[True, True],
        na_position="last"
    )

    valid = combined.dropna(subset=["throughput_fps"])

    if len(valid) > 0:
        fastest = valid.loc[valid["p50_latency_ms"].idxmin()]
        cheapest_cpu = valid[valid["hardware"] == "CPU"]
        cheapest = cheapest_cpu.loc[cheapest_cpu["throughput_fps"].idxmax()] if len(cheapest_cpu) > 0 else None

        print("\n=== BEST OPTIONS ===")
        print(f"\nFastest (lowest latency):")
        print(f"  {fastest['option']} on {fastest['hardware']}: "
              f"p50={fastest['p50_latency_ms']}ms, throughput={fastest['throughput_fps']} FPS")

        if cheapest is not None:
            print(f"\nCheapest (best CPU-only throughput):")
            print(f"  {cheapest['option']} on {cheapest['hardware']}: "
                  f"p50={cheapest['p50_latency_ms']}ms, throughput={cheapest['throughput_fps']} FPS")

    output_path = os.path.join(results_dir, "serving_options_table_final.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nFinal table saved to {output_path}")
    print(f"Total rows: {len(combined)}")
    print("\n" + combined.to_string(index=False))


if __name__ == "__main__":
    main()
