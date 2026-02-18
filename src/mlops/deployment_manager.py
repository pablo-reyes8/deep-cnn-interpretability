import argparse
import json
import shutil
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def promote_candidate(
    *,
    gate_report_path: str,
    candidate_model_path: str,
    target_model_path: str,
    state_path: str,
    deployment_history_path: str,
    backup_dir: str,
) -> dict[str, Any]:
    gate_report_file = Path(gate_report_path)
    candidate_file = Path(candidate_model_path)
    target_file = Path(target_model_path)
    state_file = Path(state_path)
    deploy_history_file = Path(deployment_history_path)
    backup_dir_path = Path(backup_dir)

    gate_report = _read_json(gate_report_file)
    gate_passed = bool(gate_report.get("gate_passed", False))
    if not gate_passed:
        raise RuntimeError("No se puede promover: quality gate no aprobado.")
    if not candidate_file.exists():
        raise FileNotFoundError(f"No existe modelo candidato: {candidate_file}")

    state = _read_json(state_file)
    if not isinstance(state, dict):
        state = {}

    ts = _utc_ts()
    backup_path = None
    previous_state_entry = None

    if target_file.exists():
        backup_dir_path.mkdir(parents=True, exist_ok=True)
        backup_name = f"{target_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target_file.suffix}"
        backup_path = backup_dir_path / backup_name
        shutil.copy2(target_file, backup_path)
        previous_state_entry = {
            "backup_path": str(backup_path),
            "metrics": state.get("current_metrics"),
            "run_id": state.get("current_run_id"),
            "promoted_at": state.get("current_promoted_at"),
        }

    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_file, target_file)

    candidate_metrics = gate_report.get("candidate", {})
    state["current_model_path"] = str(target_file)
    state["current_run_id"] = candidate_metrics.get("run_id")
    state["current_metrics"] = {
        "val_roc_auc": candidate_metrics.get("val_roc_auc"),
        "val_acc": candidate_metrics.get("val_acc"),
        "val_loss": candidate_metrics.get("val_loss"),
    }
    state["current_promoted_at"] = ts
    if previous_state_entry is not None:
        history = state.get("backups", [])
        if not isinstance(history, list):
            history = []
        history.append(previous_state_entry)
        state["backups"] = history[-20:]

    _write_json(state_file, state)

    event = {
        "timestamp_utc": ts,
        "action": "promote",
        "gate_report_path": str(gate_report_file),
        "candidate_model_path": str(candidate_file),
        "target_model_path": str(target_file),
        "backup_path": str(backup_path) if backup_path else None,
        "run_id": candidate_metrics.get("run_id"),
        "metrics": state.get("current_metrics"),
    }
    _append_jsonl(deploy_history_file, event)
    return event


def rollback_model(
    *,
    target_model_path: str,
    state_path: str,
    rollback_history_path: str,
    reason: str,
) -> dict[str, Any]:
    ts = _utc_ts()
    target_file = Path(target_model_path)
    state_file = Path(state_path)
    rollback_history_file = Path(rollback_history_path)

    state = _read_json(state_file)
    backups = state.get("backups", []) if isinstance(state, dict) else []
    if not isinstance(backups, list):
        backups = []

    rollback_success = False
    restored_backup = None
    restored_metrics = None
    restored_run_id = None

    if backups:
        candidate = backups.pop()
        backup_path = Path(candidate.get("backup_path", ""))
        if backup_path.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, target_file)
            rollback_success = True
            restored_backup = str(backup_path)
            restored_metrics = candidate.get("metrics")
            restored_run_id = candidate.get("run_id")
            state["current_model_path"] = str(target_file)
            state["current_run_id"] = restored_run_id
            state["current_metrics"] = restored_metrics
            state["current_promoted_at"] = ts
            state["backups"] = backups
            _write_json(state_file, state)

    event = {
        "timestamp_utc": ts,
        "action": "rollback",
        "reason": reason,
        "rollback_success": rollback_success,
        "target_model_path": str(target_file),
        "restored_backup_path": restored_backup,
        "restored_run_id": restored_run_id,
        "restored_metrics": restored_metrics,
    }
    _append_jsonl(rollback_history_file, event)
    return event


def parse_args():
    parser = argparse.ArgumentParser(description="Gestion de despliegue/promocion/rollback de modelo.")
    parser.add_argument("--action", type=str, required=True, choices=["promote", "rollback"])
    parser.add_argument("--gate-report-path", type=str, default="monitoring/quality_gate_report.json")
    parser.add_argument("--candidate-model-path", type=str, default="resnet101/model_trained/mlops/best_model.pth")
    parser.add_argument("--target-model-path", type=str, default="resnet101/model_trained/ResNet101.pth")
    parser.add_argument("--state-path", type=str, default="monitoring/deployment_state.json")
    parser.add_argument("--deployment-history-path", type=str, default="monitoring/deployment_history.jsonl")
    parser.add_argument("--rollback-history-path", type=str, default="monitoring/rollback_history.jsonl")
    parser.add_argument("--backup-dir", type=str, default="resnet101/model_trained/backups")
    parser.add_argument("--reason", type=str, default="automatic_rollback")
    parser.add_argument("--allow-noop-rollback", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action == "promote":
        event = promote_candidate(
            gate_report_path=args.gate_report_path,
            candidate_model_path=args.candidate_model_path,
            target_model_path=args.target_model_path,
            state_path=args.state_path,
            deployment_history_path=args.deployment_history_path,
            backup_dir=args.backup_dir,
        )
        print(f"[OK] Promocion aplicada. Modelo activo: {event.get('target_model_path')}")
        return

    event = rollback_model(
        target_model_path=args.target_model_path,
        state_path=args.state_path,
        rollback_history_path=args.rollback_history_path,
        reason=args.reason,
    )
    print(f"[OK] Rollback ejecutado. success={event.get('rollback_success')}")
    if not event.get("rollback_success") and not args.allow_noop_rollback:
        raise SystemExit(4)


if __name__ == "__main__":
    main()

