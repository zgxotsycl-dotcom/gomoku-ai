# Automation & Monitoring Enhancements

## Metric Exporter
A lightweight exporter is available at `dist/monitor/export_status.js`.

Run once to emit Prometheus metrics:
```bash
node dist/monitor/export_status.js > status.prom
```
Environment variables:
- `STATUS_PATH`: path to `status.json` (default `logs/status.json`)
- `EXPORT_FORMAT`: `prometheus` (default) or `json`

You can call this periodically via cron or a Kubernetes `CronJob` and feed the output into Prometheus/Pushgateway or any log aggregator.

## Pipeline Orchestration
- Wrap `npm run docker:gpu` in a systemd service or CI workflow so training restarts automatically after reboots.
- For multi-node setups, consider using Airflow/Prefect for self-play → distillation → evaluation scheduling. Each stage is already exposed as standalone scripts under `dist/` (`self_play_trainer.js`, `distill_student.js`, `arena.js`, `upload_model.js`).

## Monitoring checklist
1. **Live logs**: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f trainer`
2. **Fine-grained status**: `tail -f replay_buffer/*.jsonl` for sample throughput.
3. **Dashboard**: Point Prometheus/Grafana to the exporter output to visualise workers, epochs, arena winrates, and error states.
4. **Alerts**: Trigger notifications when `gomoku_error` equals 1 or arena winrate drops below target.

## Continuous Delivery
- After each promoted model, run `node dist/upload_model.js` to push to Supabase.
- The inference VM already reloads remote models based on `MODEL_CHECK_INTERVAL_MS`; ensure Supabase `latest/model.json` is updated at the end of every pipeline cycle.
- For canary deployments, add a second VM pointing to `models/candidate/model.json` and compare responses before switching `MODEL_URL`.


## Self-Play Variants
- Standard GPU pipeline: `npm run docker:gpu`
- High worker count: `npm run docker:gpu:workers` (NUM_WORKERS=16 등 상위 자원 사용)

## One-shot pipeline cycle
- 전체 루프 실행: `npm run pipeline:cycle` (self-play → distillation → arena → 업로드 순으로 실행)


## Experiment Runner
- Define experiment matrix in `experiments.json`, e.g.
  ```json
  [{"name":"deep-128","env":{"RESIDUAL_BLOCKS":"9","CONV_FILTERS":"128"}}]
  ```
- Run `npm run build && npm run experiments` to execute each configuration sequentially via the pipeline cycle.
- Set `EXPERIMENT_CONFIG_PATH` to use an alternate JSON file.


## Reporting
- Quick summary: `npm run build && npm run report:status` to print phase/self-play/arena/upload details.
- Override paths with `STATUS_PATH` or `ARENA_RESULT_PATH`.
- Combine with cron to archive daily snapshots.


## Strategy Analysis
- `npm run analyze:psq -- --dir <psq_dir>`: parses Gomocup `.psq` logs and reports first-move loss hot spots.
- Configure with `PSQ_ANALYZE_DIR` and `PSQ_ANALYZE_PLAYER1_IS_BLACK`.

## Replay Buffer Curation
- `npm run curate:replay` removes duplicate states from `replay_buffer/` and writes `curated.jsonl`.
- Use env vars `CURATE_INPUT_DIR`, `CURATE_OUTPUT`, `CURATE_MAX_SAMPLES` to control scope.

## Inference Health Monitor
- `npm run check:inference` pings the inference server (default `http://localhost:8080/health`).
- Override `CHECK_URL`, `CHECK_EXPECT_KEY`, and integrate with cron/systemd for automatic restarts.

## Hyperparameter Auto Tuning
- `npm run tune:hparams` generates a grid of experiments (`experiments.auto.json`).
- Set `TUNE_RESIDUAL_BLOCKS`, `TUNE_CONV_FILTERS`, `TUNE_LEARNING_RATES`, `TUNE_MCTS_THINK_MS`.
- `TUNE_RUN=true` runs the generated experiments immediately.


## Full Automation Runner
- `npm run build && npm run automation:auto` executes the default sequence (build → ingest batches → curate → tune → experiments → pipeline → analysis → report → health check).
- Customize by creating `automation.json` with an array of tasks `{ "name": "...", "npm": "script", "args": [...], "env": { KEY: VALUE } }`.
- Override config location using `AUTO_CONFIG_PATH`.
  - Set `AUTO_PIPELINE_TF_USE_GPU=1` if GPU DLLs are available; defaults to CPU fallback (`TF_USE_GPU=0`).
