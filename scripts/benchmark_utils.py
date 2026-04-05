"""
Shared benchmark utilities for model evaluation.

Provides reusable functions for measuring inference latency, throughput,
and model size across different execution backends (PyTorch, ONNX Runtime,
scikit-learn, FastText).
"""

import os
import time
import numpy as np


def get_model_size_mb(path):
    """Return total size of a file or directory in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e6
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e6


def benchmark_latency(predict_fn, sample_input, num_trials=100, warmup=5):
    """
    Measure single-sample inference latency.

    Args:
        predict_fn: callable that takes sample_input and returns prediction
        sample_input: single input sample
        num_trials: number of inference runs
        warmup: number of warmup runs before timing

    Returns:
        dict with median_ms, p95_ms, p99_ms, throughput_fps, raw_latencies
    """
    for _ in range(warmup):
        predict_fn(sample_input)

    latencies = []
    for _ in range(num_trials):
        start = time.perf_counter()
        predict_fn(sample_input)
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)
    return {
        "median_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "p99_ms": float(np.percentile(latencies, 99) * 1000),
        "throughput_fps": float(num_trials / np.sum(latencies)),
        "raw_latencies": latencies,
    }


def benchmark_batch_throughput(predict_fn, batch_input, num_batches=50, warmup=3):
    """
    Measure batch inference throughput.

    Args:
        predict_fn: callable that takes batch_input and returns predictions
        batch_input: a batch of inputs
        num_batches: number of batch runs
        warmup: warmup iterations

    Returns:
        dict with batch_fps and batch_size
    """
    for _ in range(warmup):
        predict_fn(batch_input)

    batch_times = []
    for _ in range(num_batches):
        start = time.perf_counter()
        predict_fn(batch_input)
        batch_times.append(time.perf_counter() - start)

    batch_times = np.array(batch_times)

    if isinstance(batch_input, dict):
        batch_size = len(next(iter(batch_input.values())))
    elif isinstance(batch_input, (list, tuple)):
        batch_size = len(batch_input)
    elif hasattr(batch_input, "shape"):
        batch_size = batch_input.shape[0]
    else:
        batch_size = 1

    return {
        "batch_fps": float(batch_size * num_batches / np.sum(batch_times)),
        "batch_size": batch_size,
    }


def benchmark_ort_session(ort_session, sample_input_dict, batch_input_dict,
                          num_trials=100, num_batches=50):
    """
    Full benchmark for an ONNX Runtime session.

    Args:
        ort_session: onnxruntime.InferenceSession
        sample_input_dict: dict mapping input names to numpy arrays (single sample)
        batch_input_dict: dict mapping input names to numpy arrays (batch)
        num_trials: trials for latency measurement
        num_batches: batches for throughput measurement

    Returns:
        dict with all metrics
    """
    def predict_single(inp):
        return ort_session.run(None, inp)

    latency = benchmark_latency(predict_single, sample_input_dict, num_trials)
    batch = benchmark_batch_throughput(predict_single, batch_input_dict, num_batches)

    return {
        "providers": ort_session.get_providers(),
        **latency,
        **batch,
    }


def print_benchmark_results(results, model_name="", config_name="", model_size_mb=None):
    """Pretty-print benchmark results."""
    header = f"=== {model_name} / {config_name} ==="
    print(header)
    if model_size_mb is not None:
        print(f"  Model Size on Disk: {model_size_mb:.2f} MB")
    if "providers" in results:
        print(f"  Execution Providers: {results['providers']}")
    print(f"  Inference Latency (single sample, median):  {results['median_ms']:.2f} ms")
    print(f"  Inference Latency (single sample, p95):     {results['p95_ms']:.2f} ms")
    print(f"  Inference Latency (single sample, p99):     {results['p99_ms']:.2f} ms")
    print(f"  Single-Sample Throughput:                    {results['throughput_fps']:.2f} FPS")
    if "batch_fps" in results:
        print(f"  Batch Throughput (batch_size={results.get('batch_size', '?')}): {results['batch_fps']:.2f} FPS")
    print()


def collect_result_row(model_name, config_name, hardware, model_size_mb, results):
    """Return a dict suitable for building the final comparison table."""
    return {
        "option": f"{model_name}_{config_name}",
        "model": model_name,
        "config": config_name,
        "hardware": hardware,
        "model_size_mb": round(model_size_mb, 2) if model_size_mb else None,
        "p50_latency_ms": round(results["median_ms"], 2),
        "p95_latency_ms": round(results["p95_ms"], 2),
        "p99_latency_ms": round(results["p99_ms"], 2),
        "throughput_fps": round(results["throughput_fps"], 2),
        "batch_throughput_fps": round(results.get("batch_fps", 0), 2),
    }
