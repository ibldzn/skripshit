#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO classification ONNX model on edge device."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path ke file model ONNX, contoh: models/yolov8n-best.onnx",
    )

    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Path folder test dataset. Default: dataset/test",
    )

    parser.add_argument(
        "--classes",
        default="classes.txt",
        help="Path file classes.txt. Urutan class harus sama dengan output model.",
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix nama output. Kalau kosong, otomatis dari nama model.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Ukuran input model. Default: 224",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Jumlah gambar untuk warm-up. Default: 10",
    )

    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=4,
        help="Jumlah intra_op_num_threads ONNX Runtime. Default: 4",
    )

    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=1,
        help="Jumlah inter_op_num_threads ONNX Runtime. Default: 1",
    )

    parser.add_argument(
        "--memory-sample-interval",
        type=float,
        default=0.01,
        help="Interval sampling memori dalam detik. Default: 0.01",
    )

    return parser.parse_args()


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def load_class_names(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        raise FileNotFoundError(f"File class tidak ditemukan: {classes_path}")

    with classes_path.open("r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if not classes:
        raise RuntimeError(f"File class kosong: {classes_path}")

    return classes


def collect_images(test_dir: Path) -> list[tuple[Path, str]]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Folder test tidak ditemukan: {test_dir}")

    images: list[tuple[Path, str]] = []

    for class_dir in sorted(test_dir.iterdir(), key=lambda p: p.name):
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.iterdir(), key=lambda p: p.name):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((img_path, class_dir.name))

    if not images:
        raise RuntimeError(f"Tidak ada gambar ditemukan di {test_dir}")

    return images


def validate_dataset_classes(class_names: list[str], images: list[tuple[Path, str]]) -> None:
    dataset_classes = sorted({true_class for _, true_class in images})
    class_file_classes = sorted(class_names)

    if dataset_classes != class_file_classes:
        print("\n[WARNING] Class pada folder dataset tidak sama dengan classes.txt")
        print("Dataset classes :", dataset_classes)
        print("classes.txt     :", class_file_classes)
        print(
            "Pastikan nama folder test sama dengan nama class output model. "
            "Kalau tidak sama, akurasi bisa salah.\n"
        )


def preprocess_image(image_path: Path, img_size: int) -> np.ndarray:
    """
    Preprocessing untuk YOLO classification ONNX:
    - Baca citra
    - Konversi RGB
    - Resize ke img_size x img_size
    - Normalisasi 0..1
    - HWC -> CHW
    - Tambah batch dimension
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr


def to_probabilities(output: np.ndarray) -> np.ndarray:
    """
    Beberapa model mengeluarkan logits, sebagian bisa sudah probability.
    Kalau output sudah tampak seperti probability, pakai langsung.
    Kalau belum, apply softmax.
    """
    x = output.astype(np.float64)

    if x.ndim > 1:
        x = x.reshape(-1)

    looks_like_prob = (
        np.all(x >= 0.0)
        and np.all(x <= 1.0)
        and np.isclose(np.sum(x), 1.0, atol=1e-3)
    )

    if looks_like_prob:
        return x

    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class MemorySampler:
    """
    Sampling RSS proses secara periodik untuk estimasi peak memory.
    Ini bukan hardware-level peak mutlak, tapi jauh lebih valid daripada
    hanya memory before dan memory after.
    """

    def __init__(self, process: psutil.Process, interval: float = 0.01) -> None:
        self.process = process
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_mb: list[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            rss_mb = self.process.memory_info().rss / (1024 * 1024)
            self.samples_mb.append(rss_mb)
            time.sleep(self.interval)

    @property
    def peak_mb(self) -> float:
        return max(self.samples_mb) if self.samples_mb else 0.0

    @property
    def average_mb(self) -> float:
        return statistics.mean(self.samples_mb) if self.samples_mb else 0.0


def create_session(
    model_path: Path,
    intra_op_threads: int,
    inter_op_threads: int,
) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_op_threads
    session_options.inter_op_num_threads = inter_op_threads

    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "avg": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
        }

    sorted_values = sorted(values)
    p95_idx = int(round(0.95 * (len(sorted_values) - 1)))

    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": sorted_values[p95_idx],
    }


def build_confusion_matrix(
    class_names: list[str],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix = defaultdict(lambda: defaultdict(int))

    for row in rows:
        matrix[row["true_class"]][row["predicted_class"]] += 1

    output_rows: list[dict[str, Any]] = []

    for true_class in class_names:
        out = {"true_class": true_class}
        for pred_class in class_names:
            out[pred_class] = matrix[true_class][pred_class]
        output_rows.append(out)

    return output_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    test_dir = Path(args.test_dir)
    classes_path = Path(args.classes)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = model_path.stem

    per_image_csv = Path(f"{output_prefix}_per_image.csv")
    confusion_csv = Path(f"{output_prefix}_confusion_matrix.csv")
    summary_json = Path(f"{output_prefix}_summary.json")

    class_names = load_class_names(classes_path)
    images = collect_images(test_dir)
    validate_dataset_classes(class_names, images)

    print("=== Benchmark Configuration ===")
    print("Model                 :", model_path)
    print("Test dir              :", test_dir)
    print("Classes               :", len(class_names))
    print("Total images          :", len(images))
    print("Image size            :", args.img_size)
    print("Warm-up               :", args.warmup)
    print("Provider              : CPUExecutionProvider")
    print("intra_op_num_threads  :", args.intra_op_threads)
    print("inter_op_num_threads  :", args.inter_op_threads)

    session = create_session(
        model_path=model_path,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    process = psutil.Process()
    cpu_count = psutil.cpu_count(logical=True) or 1

    # Warm-up
    warmup_images = images[: min(args.warmup, len(images))]
    for img_path, _ in warmup_images:
        x = preprocess_image(img_path, args.img_size)
        _ = session.run([output_name], {input_name: x})

    rows: list[dict[str, Any]] = []

    preprocess_latencies: list[float] = []
    inference_latencies: list[float] = []
    postprocess_latencies: list[float] = []
    total_latencies: list[float] = []

    correct = 0

    rss_before_mb = process.memory_info().rss / (1024 * 1024)
    cpu_times_before = process.cpu_times()
    wall_start = time.perf_counter()

    memory_sampler = MemorySampler(
        process=process,
        interval=args.memory_sample_interval,
    )
    memory_sampler.start()

    try:
        for img_path, true_class in images:
            total_start_ms = now_ms()

            preprocess_start_ms = now_ms()
            x = preprocess_image(img_path, args.img_size)
            preprocess_end_ms = now_ms()

            inference_start_ms = now_ms()
            outputs = session.run([output_name], {input_name: x})
            inference_end_ms = now_ms()

            postprocess_start_ms = now_ms()
            probs = to_probabilities(outputs[0][0])
            pred_idx = int(np.argmax(probs))

            if pred_idx >= len(class_names):
                raise RuntimeError(
                    f"Output model menghasilkan index {pred_idx}, "
                    f"tetapi jumlah class hanya {len(class_names)}."
                )

            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])
            postprocess_end_ms = now_ms()

            total_end_ms = now_ms()

            preprocess_ms = preprocess_end_ms - preprocess_start_ms
            inference_ms = inference_end_ms - inference_start_ms
            postprocess_ms = postprocess_end_ms - postprocess_start_ms
            total_ms = total_end_ms - total_start_ms

            is_correct = pred_class == true_class
            correct += int(is_correct)

            preprocess_latencies.append(preprocess_ms)
            inference_latencies.append(inference_ms)
            postprocess_latencies.append(postprocess_ms)
            total_latencies.append(total_ms)

            rows.append(
                {
                    "image_path": str(img_path),
                    "true_class": true_class,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "correct": int(is_correct),
                    "preprocess_ms": preprocess_ms,
                    "inference_ms": inference_ms,
                    "postprocess_ms": postprocess_ms,
                    "total_pipeline_ms": total_ms,
                }
            )

    finally:
        memory_sampler.stop()

    wall_end = time.perf_counter()
    cpu_times_after = process.cpu_times()

    rss_after_mb = process.memory_info().rss / (1024 * 1024)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    wall_time_sec = wall_end - wall_start
    cpu_time_delta_sec = (
        (cpu_times_after.user + cpu_times_after.system)
        - (cpu_times_before.user + cpu_times_before.system)
    )

    # Bisa >100% kalau proses pakai beberapa core.
    process_cpu_percent_total = (
        (cpu_time_delta_sec / wall_time_sec) * 100 if wall_time_sec > 0 else 0.0
    )

    # Dinormalisasi terhadap jumlah logical CPU.
    process_cpu_percent_normalized = process_cpu_percent_total / cpu_count

    accuracy = correct / len(images)
    inference_summary = summarize(inference_latencies)
    preprocess_summary = summarize(preprocess_latencies)
    postprocess_summary = summarize(postprocess_latencies)
    total_summary = summarize(total_latencies)

    fps_inference_only = (
        1000.0 / inference_summary["avg"] if inference_summary["avg"] > 0 else 0.0
    )

    fps_total_pipeline = (
        1000.0 / total_summary["avg"] if total_summary["avg"] > 0 else 0.0
    )

    confusion_rows = build_confusion_matrix(class_names, rows)

    write_csv(per_image_csv, rows)
    write_csv(confusion_csv, confusion_rows)

    summary = {
        "model": str(model_path),
        "model_size_mb": model_size_mb,
        "test_dir": str(test_dir),
        "total_images": len(images),
        "correct": correct,
        "accuracy": accuracy,
        "img_size": args.img_size,
        "warmup": args.warmup,
        "provider": "CPUExecutionProvider",
        "intra_op_num_threads": args.intra_op_threads,
        "inter_op_num_threads": args.inter_op_threads,
        "batch_size": 1,
        "preprocess_ms": preprocess_summary,
        "inference_ms": inference_summary,
        "postprocess_ms": postprocess_summary,
        "total_pipeline_ms": total_summary,
        "fps_inference_only": fps_inference_only,
        "fps_total_pipeline": fps_total_pipeline,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_delta_mb": rss_after_mb - rss_before_mb,
        "rss_peak_sampled_mb": memory_sampler.peak_mb,
        "rss_average_sampled_mb": memory_sampler.average_mb,
        "wall_time_sec": wall_time_sec,
        "process_cpu_percent_total": process_cpu_percent_total,
        "process_cpu_percent_normalized": process_cpu_percent_normalized,
        "output_files": {
            "per_image_csv": str(per_image_csv),
            "confusion_matrix_csv": str(confusion_csv),
            "summary_json": str(summary_json),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Benchmark Result ===")
    print("Model                      :", model_path)
    print("Model size MB              :", f"{model_size_mb:.4f}")
    print("Total images               :", len(images))
    print("Correct                    :", correct)
    print("Accuracy                   :", f"{accuracy:.6f}")

    print("\n--- Inference Only ---")
    print("Avg inference ms           :", f"{inference_summary['avg']:.6f}")
    print("Median inference ms        :", f"{inference_summary['median']:.6f}")
    print("Min inference ms           :", f"{inference_summary['min']:.6f}")
    print("Max inference ms           :", f"{inference_summary['max']:.6f}")
    print("P95 inference ms           :", f"{inference_summary['p95']:.6f}")
    print("FPS inference only         :", f"{fps_inference_only:.6f}")

    print("\n--- Total Pipeline: read + preprocess + inference + postprocess ---")
    print("Avg total ms               :", f"{total_summary['avg']:.6f}")
    print("Median total ms            :", f"{total_summary['median']:.6f}")
    print("Min total ms               :", f"{total_summary['min']:.6f}")
    print("Max total ms               :", f"{total_summary['max']:.6f}")
    print("P95 total ms               :", f"{total_summary['p95']:.6f}")
    print("FPS total pipeline         :", f"{fps_total_pipeline:.6f}")

    print("\n--- Memory ---")
    print("RSS before MB              :", f"{rss_before_mb:.4f}")
    print("RSS after MB               :", f"{rss_after_mb:.4f}")
    print("RSS delta MB               :", f"{rss_after_mb - rss_before_mb:.4f}")
    print("RSS peak sampled MB        :", f"{memory_sampler.peak_mb:.4f}")
    print("RSS average sampled MB     :", f"{memory_sampler.average_mb:.4f}")

    print("\n--- CPU ---")
    print("Wall time sec              :", f"{wall_time_sec:.4f}")
    print("Process CPU % total        :", f"{process_cpu_percent_total:.4f}")
    print("Process CPU % normalized   :", f"{process_cpu_percent_normalized:.4f}")

    print("\n--- Output Files ---")
    print("Per-image CSV              :", per_image_csv)
    print("Confusion matrix CSV       :", confusion_csv)
    print("Summary JSON               :", summary_json)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark dihentikan.")
        sys.exit(130)
