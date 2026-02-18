import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _candidate_from_metrics(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": metrics_payload.get("run_id"),
        "best_epoch": metrics_payload.get("best_epoch"),
        "val_roc_auc": _safe_float(metrics_payload.get("best_val_roc_auc"), 0.0),
        "val_acc": _safe_float(metrics_payload.get("best_val_acc"), 0.0),
        "val_loss": _safe_float(metrics_payload.get("best_val_loss"), 1e9),
    }


def run_quality_gate(
    *,
    candidate_metrics_path: str,
    candidate_model_path: str,
    state_path: str,
    report_path: str,
    history_path: str,
    min_val_roc_auc: float = 0.85,
    min_val_acc: float = 0.80,
    max_val_loss: float = 0.70,
    min_delta_roc_auc: float = -0.01,
    min_delta_acc: float = -0.02,
    max_delta_loss: float = 0.05,
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()

    cand_metrics_file = Path(candidate_metrics_path)
    cand_model_file = Path(candidate_model_path)
    state_file = Path(state_path)
    report_file = Path(report_path)
    history_file = Path(history_path)

    metrics_payload = _read_json(cand_metrics_file)
    candidate = _candidate_from_metrics(metrics_payload)
    candidate["model_path"] = str(cand_model_file)
    candidate["metrics_path"] = str(cand_metrics_file)

    state = _read_json(state_file)
    baseline = state.get("current_metrics", {}) if isinstance(state, dict) else {}

    baseline_exists = bool(baseline)
    base_roc = _safe_float(baseline.get("val_roc_auc"), 0.0)
    base_acc = _safe_float(baseline.get("val_acc"), 0.0)
    base_loss = _safe_float(baseline.get("val_loss"), 1e9)

    delta_roc = candidate["val_roc_auc"] - base_roc if baseline_exists else None
    delta_acc = candidate["val_acc"] - base_acc if baseline_exists else None
    delta_loss = candidate["val_loss"] - base_loss if baseline_exists else None

    checks = {
        "candidate_model_exists": cand_model_file.exists(),
        "abs_min_roc_auc": candidate["val_roc_auc"] >= min_val_roc_auc,
        "abs_min_acc": candidate["val_acc"] >= min_val_acc,
        "abs_max_loss": candidate["val_loss"] <= max_val_loss,
    }

    if baseline_exists:
        checks.update(
            {
                "delta_min_roc_auc": (delta_roc is not None and delta_roc >= min_delta_roc_auc),
                "delta_min_acc": (delta_acc is not None and delta_acc >= min_delta_acc),
                "delta_max_loss": (delta_loss is not None and delta_loss <= max_delta_loss),
            }
        )
    else:
        checks.update(
            {
                "delta_min_roc_auc": True,
                "delta_min_acc": True,
                "delta_max_loss": True,
            }
        )

    gate_passed = all(bool(v) for v in checks.values())
    reasons = [k for k, v in checks.items() if not bool(v)]

    report = {
        "timestamp_utc": ts,
        "gate_passed": gate_passed,
        "failed_checks": reasons,
        "candidate": candidate,
        "baseline": {
            "exists": baseline_exists,
            "run_id": baseline.get("run_id") if baseline_exists else None,
            "val_roc_auc": base_roc if baseline_exists else None,
            "val_acc": base_acc if baseline_exists else None,
            "val_loss": base_loss if baseline_exists else None,
        },
        "deltas": {
            "delta_roc_auc": delta_roc,
            "delta_acc": delta_acc,
            "delta_loss": delta_loss,
        },
        "thresholds": {
            "min_val_roc_auc": min_val_roc_auc,
            "min_val_acc": min_val_acc,
            "max_val_loss": max_val_loss,
            "min_delta_roc_auc": min_delta_roc_auc,
            "min_delta_acc": min_delta_acc,
            "max_delta_loss": max_delta_loss,
        },
        "checks": checks,
        "state_path": str(state_file),
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _append_jsonl(history_file, report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Quality gate post-entrenamiento.")
    parser.add_argument("--candidate-metrics-path", type=str, default="resnet101/model_trained/mlops/metrics.json")
    parser.add_argument("--candidate-model-path", type=str, default="resnet101/model_trained/mlops/best_model.pth")
    parser.add_argument("--state-path", type=str, default="monitoring/deployment_state.json")
    parser.add_argument("--report-path", type=str, default="monitoring/quality_gate_report.json")
    parser.add_argument("--history-path", type=str, default="monitoring/quality_gate_history.jsonl")
    parser.add_argument("--min-val-roc-auc", type=float, default=0.85)
    parser.add_argument("--min-val-acc", type=float, default=0.80)
    parser.add_argument("--max-val-loss", type=float, default=0.70)
    parser.add_argument("--min-delta-roc-auc", type=float, default=-0.01)
    parser.add_argument("--min-delta-acc", type=float, default=-0.02)
    parser.add_argument("--max-delta-loss", type=float, default=0.05)
    parser.add_argument("--exit-code-on-fail", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_quality_gate(
        candidate_metrics_path=args.candidate_metrics_path,
        candidate_model_path=args.candidate_model_path,
        state_path=args.state_path,
        report_path=args.report_path,
        history_path=args.history_path,
        min_val_roc_auc=args.min_val_roc_auc,
        min_val_acc=args.min_val_acc,
        max_val_loss=args.max_val_loss,
        min_delta_roc_auc=args.min_delta_roc_auc,
        min_delta_acc=args.min_delta_acc,
        max_delta_loss=args.max_delta_loss,
    )
    print(f"[OK] Quality gate report: {args.report_path}")
    print(f"[OK] Gate passed: {report.get('gate_passed')}")
    if not report.get("gate_passed", False):
        sys.exit(args.exit_code_on_fail)
    sys.exit(0)


if __name__ == "__main__":
    main()

