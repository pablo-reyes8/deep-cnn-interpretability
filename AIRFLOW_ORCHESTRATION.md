# Airflow Orchestration (MLOps)

## Qué incluye

- DAG: `airflow/dags/resnet101_mlops_orchestrator.py`
- Quality gate post-entrenamiento: `src/mlops/quality_gate.py`
- Gestor de promotion/rollback: `src/mlops/deployment_manager.py`
- Health check de comportamiento: `src/mlops/evaluate_model_health.py`
- Stack Airflow: `docker-compose.airflow.yml` + `docker/airflow.Dockerfile`

## Lógica del DAG

1. Bootstrap (si no hay modelo en producción):
   - `ingestion -> training -> quality_gate`.
   - Si pasa gate: `promote -> deploy -> post_deploy_health`.
   - Si falla gate o post-deploy sale degradado: `rollback`.
2. Monitoreo continuo:
   - `detect_drift`.
   - `evaluate_model_health`.
3. Reentrenamiento condicional:
   - Si hay drift o degradación: `training -> quality_gate`.
   - Si pasa gate: `promote -> rollout`.
   - Si falla o se degrada post-rollout: `rollback` automático.
4. Trazabilidad:
   - `quality_gate_history.jsonl`
   - `deployment_history.jsonl`
   - `rollback_history.jsonl`
   - `orchestration_report.json`

## Levantar Airflow

```bash
docker compose -f docker-compose.airflow.yml up -d --build
```

Airflow UI:

- `http://localhost:8080`

## DAG

- Nombre: `resnet101_mlops_orchestrator`
- Schedule: cada 2 horas (`0 */2 * * *`)

## Archivos de monitoreo usados por el DAG

- `monitoring/inference_events.jsonl` (inferencia real)
- `monitoring/feedback_events.jsonl` (opcional con etiquetas reales)
- `monitoring/drift_report.json`
- `monitoring/model_health_report.json`
- `monitoring/quality_gate_report_bootstrap.json`
- `monitoring/quality_gate_report_retrain.json`
- `monitoring/deployment_state.json`
- `monitoring/deployment_history.jsonl`
- `monitoring/rollback_history.jsonl`

## Formato recomendado para feedback

```json
{"timestamp_utc":"2026-02-18T20:00:00+00:00","prediction":"cat","true_label":"dog"}
```

o

```json
{"timestamp_utc":"2026-02-18T20:00:00+00:00","prediction_correct":false}
```
