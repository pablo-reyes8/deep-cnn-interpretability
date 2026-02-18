import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_reference_stats(reference_stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(reference_stats_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de referencia: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    loc = np.asarray(payload.get("loc", []), dtype=np.float64)
    scale = np.asarray(payload.get("scale", []), dtype=np.float64)
    if loc.size != 3 or scale.size != 3:
        raise ValueError("El archivo de referencia debe tener 'loc' y 'scale' con 3 canales.")
    return loc, scale


def load_inference_events(log_path: str, window_size: int) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines[-window_size:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _extract_matrix(events: list[dict[str, Any]], field: str) -> np.ndarray:
    rows = []
    for e in events:
        v = e.get(field)
        if isinstance(v, list) and len(v) == 3:
            try:
                rows.append([float(x) for x in v])
            except (TypeError, ValueError):
                continue
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _extract_confidence(events: list[dict[str, Any]]) -> np.ndarray:
    out = []
    for e in events:
        try:
            out.append(float(e.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
    if not out:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def analyze_drift(
    *,
    reference_stats_path: str,
    inference_log_path: str,
    window_size: int = 300,
    min_samples: int = 50,
    mean_shift_threshold: float = 0.35,
    scale_shift_threshold: float = 0.25,
    min_avg_confidence: float = 0.60,
) -> Dict[str, Any]:
    ref_loc, ref_scale = load_reference_stats(reference_stats_path)
    events = load_inference_events(inference_log_path, window_size=window_size)

    channel_means = _extract_matrix(events, "raw_channel_mean")
    channel_stds = _extract_matrix(events, "raw_channel_std")
    confidence = _extract_confidence(events)

    sample_count = int(min(len(channel_means), len(channel_stds)))
    if sample_count < min_samples:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_data",
            "drift_detected": False,
            "samples": sample_count,
            "min_samples_required": min_samples,
            "window_size": window_size,
        }

    means_window = channel_means[:sample_count]
    stds_window = channel_stds[:sample_count]
    confidence_window = confidence[:sample_count] if confidence.size else np.empty((0,))

    curr_loc = means_window.mean(axis=0)
    curr_scale = stds_window.mean(axis=0)

    eps = 1e-8
    mean_shift_sigma = np.abs(curr_loc - ref_loc) / np.maximum(ref_scale, eps)
    scale_shift_ratio = np.abs(curr_scale / np.maximum(ref_scale, eps) - 1.0)
    avg_conf = float(confidence_window.mean()) if confidence_window.size else 0.0

    mean_flag = bool(np.any(mean_shift_sigma > mean_shift_threshold))
    scale_flag = bool(np.any(scale_shift_ratio > scale_shift_threshold))
    conf_flag = bool(avg_conf < min_avg_confidence) if confidence_window.size else False
    drift_detected = bool(mean_flag or scale_flag or conf_flag)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "drift_detected": drift_detected,
        "samples": sample_count,
        "window_size": window_size,
        "reference": {
            "loc": ref_loc.tolist(),
            "scale": ref_scale.tolist(),
        },
        "current": {
            "loc": curr_loc.tolist(),
            "scale": curr_scale.tolist(),
            "avg_confidence": avg_conf,
        },
        "thresholds": {
            "mean_shift_sigma": mean_shift_threshold,
            "scale_shift_ratio": scale_shift_threshold,
            "min_avg_confidence": min_avg_confidence,
        },
        "signals": {
            "mean_shift_sigma": mean_shift_sigma.tolist(),
            "scale_shift_ratio": scale_shift_ratio.tolist(),
            "low_confidence": conf_flag,
        },
        "flags": {
            "mean_shift": mean_flag,
            "scale_shift": scale_flag,
            "confidence_shift": conf_flag,
        },
    }

