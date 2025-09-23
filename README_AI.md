AI Pipeline & Inference – Quick Start

Prerequisites
- Node.js 20+
- Trained model at `gomoku_model_prod/model.json` (exists in repo)
- Optional: Docker if you want to run the pipeline in a container

Build
- Install deps: `npm install`
- Compile: `npm run build`

Run Local Inference Server
- Start: `node dist/server.js`
- Env vars (optional):
  - `PORT` default 8080
  - `MODEL_PATH` default `gomoku_model_prod/model.json` (local file)
  - `MODEL_URL` if set (e.g., Supabase public URL), server will load remotely and auto‑reload on ETag change
  - `MODEL_CHECK_INTERVAL_MS` default 300000
  - `EARLY_GAME_MOVES` `EARLY_GAME_THINK_TIME` `MID_GAME_THINK_TIME` `LATE_GAME_THINK_TIME`
- Request:
  POST http://localhost:8080/get-move
  { "board": [[...15x15...]], "player": "black", "moves": [[r,c], ...] }

Run Self-Play Pipeline
- One-shot (install+build+run): `npm run start:one`
- Start (build+run): `npm start`
- Direct: `node dist/start_pipeline.js`
- Env vars (optional):
  - `NUM_WORKERS` default 4
  - `REPLAY_BUFFER_DIR` default `replay_buffer`
  - `PROD_MODEL_DIR` default `gomoku_model_prod`
  - `PAST_MODELS_DIR` default `past_models`
  - `SELF_PLAY_DURATION_MS` default 1800000
  - `MCTS_THINK_TIME_MS` default 4000
  - `EXPLORATION_MOVES` default 15
  - `RUN_DISTILLATION` default `true` (run KD after data gen)
  - `UPLOAD_MODEL_AFTER` default `true` (upload to Supabase after pipeline)
  - `AUTO_GENERATE_OPENING_BOOK` default `true` (build book after upload)
  - `IMPORT_OPENING_BOOK` default `true` (import book into Supabase)
  - `PIPELINE_CYCLES` default `1` (number of full cycles)
  - `PIPELINE_INTERVAL_MS` default `0` (delay between cycles)
- Output: JSONL files under `replay_buffer/`

Real-Time Training Status
- Web UI: open http://localhost:8090 while the pipeline runs.
- API: GET http://localhost:8090/status returns the current JSON status (from `logs/status.json`).
- Standalone status server (without running the whole pipeline):
  1) `npm run build`
  2) `node dist/status_server.js` (or `npm run start:status`)
  3) Open http://localhost:8090
- Customize:
  - `STATUS_PATH` to point to a different `status.json`
  - `STATUS_PORT` and `STATUS_HOST` to change bind address

Upload Trained Model to Supabase Storage
- Script: `node dist/upload_model.js`
- Env vars:
  - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (required)
  - `SUPABASE_BUCKET` default `models`
  - `SUPABASE_MODEL_PREFIX` default `gomoku_model` (files uploaded under this path)
  - `MODEL_DIR` default `gomoku_model_prod`
  - `PUBLIC_CACHE_CONTROL` default `public, max-age=60`
- Pipeline auto-upload: set `UPLOAD_MODEL_AFTER=true` before running `dist/start_pipeline.js`

Distillation Training (Knowledge Distillation)
- Script: `node dist/distill_student.js`
- Uses JSONL from `replay_buffer/` with fields: `state`, `player`, `mcts_policy`, `teacher_policy`, `teacher_value`, `final_value`.
- Env vars:
  - `BATCH_SIZE` default 64, `EPOCHS` default 4, `STEPS_PER_EPOCH` default 4000
  - `LEARNING_RATE` default 5e-4
  - `TEACHER_TEMP` default 1.5 (soften teacher policy)
  - `ALPHA_TEACHER_POLICY` default 0.7 (mix teacher vs MCTS policy)
  - `BETA_TEACHER_VALUE` default 0.7 (mix teacher vs final value)

Docker (optional)
-- Build: `docker build -t gomoku-ai .`
-- Run: `docker run --rm -p 8080:8080 -p 8090:8090 gomoku-ai` (starts pipeline+status server by default)
-- Compose (CPU): `npm run docker`
-- Compose (GPU): `npm run docker:gpu`

Docker Compose (recommended)
- Start: `./scripts/start_docker.sh` (Linux/macOS) or `./scripts/start_docker.ps1` (Windows)
- Checks:
  - Follow logs: `docker compose logs -f`
  - Persisted data in volumes: `replay_buffer`, `prod_models`, `past_models`, `logs`
- Env vars:
  - Provide `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` via `docker compose` environment or rely on `gomoku-app-v2/.env.local` auto-loading.
  - Default pipeline uses: `BOARD_SIZE=15`, `FOREVER=true`, `PIPELINE_CYCLES=0`.

GPU Option
- Default image uses CPU (`@tensorflow/tfjs-node`). GPU 가속 실행:
  - 사전: NVIDIA 드라이버 + NVIDIA Container Toolkit 설치
  - 빌드/실행(GPU): `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`
  - 컨테이너는 `Dockerfile.gpu`를 사용해 `@tensorflow/tfjs-node-gpu`를 설치/사용합니다.
  - 런타임 ENV: `TF_FORCE_GPU_ALLOW_GROWTH=1`

Arena/Evaluations Table
- A migration adds `public.ai_model_evaluations` to store arena results.
- See `supabase/migrations/20250914193000_create_ai_model_evaluations.sql` and ensure it is applied to your database.

Notes
- If GPU libs are not available, consider switching `@tensorflow/tfjs-node-gpu` to `@tensorflow/tfjs-node` in `package.json` for CPU inference.

Opening Book
- Local server loads `opening_book.json` at repo root and performs symmetry-canonical matching.
- Build/expand book with NN‑MCTS:
  - `node dist/build_opening_book.js`
  - Env: `BOOK_DEPTH` (default 3), `BOOK_BRANCHING` (default 4), `BOOK_THINK_TIME` (default 2000ms)
  - Output: `opening_book_generated.json`
- Import to Supabase (canonicalized): call Edge Function `import-opening-book` with JSON body from the generated file.
  - Or run `node dist/import_opening_book.js` with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.

Rules & Time Control
- Opening rule: set `OPENING_RULE=swap2` (default). Self-play and arena start from a Swap2-legal opening position. Set to `freestyle` to disable.
- Time control: server supports `TIME_CONTROL` like `5+1` as a fallback when the client does not send `timeLeftMs`/`turnEndsAt`. For precise time usage, pass `timeLeftMs` per turn; the server allocates a safe fraction dynamically.

Swap2 Negotiation (Advanced)
- Server env:
  - `SWAP2_ROLLOUT_MS` (default 500): per-option shallow rollout budget in ms
  - `SWAP2_ROLLOUT_PLIES` (default 3): rollout depth in plies
 - Candidate shaping:
   - `SWAP2_SYM_AVG` (default 8): avg policy over symmetries {1,4,8}
   - `SWAP2_DIST_LAMBDA` (default 0.0): early distance penalty strength (exp decay)
   - `SWAP2_DIST_PHASE_MOVES` (default 8): apply penalty while stones ≤ this

Tactical Boosting & Hybrid Priors
- Root boosts: `BOOST_CREATE`, `BOOST_BLOCK`, `BOOST_OPEN3_ROOT`, `BOOST_OPEN3_ROOT_BLOCK`, `BOOST_FOUR_ROOT`, `BOOST_FOUR_ROOT_BLOCK`, `BOOST_CONN3_ROOT`, `BOOST_CONN3_ROOT_BLOCK`, `BOOST_LINK_ROOT`
- Child boosts: `BOOST_CREATE_CHILD`, `BOOST_BLOCK_CHILD`, `BOOST_OPEN3_CHILD`, `BOOST_OPEN3_BLOCK_CHILD`, `BOOST_CONN3_CHILD`, `BOOST_CONN3_BLOCK_CHILD`, `BOOST_LINK_CHILD`, `BOOST_INSTANT_WIN_CHILD`, `BOOST_BLOCK_IMMEDIATE_CHILD`
- TT+NN priors:
  - `TT_PRIOR_MIX` at root, `CHILD_TT_PRIOR_MIX` at children

Auto Tuning (Arena)
- Enable: `TUNE_PARAMS_ON_ARENA=true` (default), step size `TUNE_LR` (default 0.03)
- After each arena, tactical boosts and `CHILD_TT_PRIOR_MIX` are nudged based on winrate vs threshold.
  - `SWAP2_CAND_W_TOPK` (default 16): top‑K White candidates by policy when evaluating Option 2/3
  - `SWAP2_CAND_B_TOPK` (default 12): top‑K Black candidates for Option 3
  - `SWAP2_USE_PATTERNS` (default true): merge tactical candidates (immediate wins/blocks/open‑fours) with policy top‑k
- Internals: the negotiator evaluates Option 1/2/3 using a short NN‑guided rollout (with small caches) to better approximate future outcomes and chooses the best from the second player’s perspective. In Option 3, the first player’s color choice is simulated (compare vW vs vB); colors are swapped if Black yields higher advantage.

MCTS Enhancements
- Transposition table: set `TT_CAP` (default 20000) to control capacity.
- Child-level threat boost: `BOOST_CREATE_CHILD` (default 1.3), `BOOST_BLOCK_CHILD` (default 1.2).

Swap2 HTTP Helpers (for AI-vs-Player / Online)
- POST `/swap2/propose` body: `{ board }` → returns `{ board }` with an initial B-W-B triple proposed near center (AI가 선공일 때 사용).
- POST `/swap2/second` body: `{ board }` → returns `{ board, toMove, swapColors }` with the second player's best option chosen via model value. 이 엔드포인트는 “AI가 후공”일 때 선택 절차를 자동화합니다.
- 로컬 P vs Player 및 비밀방 모드에서는 위 엔드포인트를 호출하지 않으면 Swap2가 적용되지 않습니다.

Gumroad Webhook (Supporters)
- Supabase Edge Function: `supabase/functions/gumroad-webhook-handler/index.ts`
- Env:
  - `GUMROAD_PRODUCT_ID_SUPPORTER` (제품 ID)
  - `GUMROAD_WEBHOOK_SECRET` (옵션: HMAC 서명 검증)
- Gumroad 설정:
  - Ping/Webhook URL: `https://<SUPABASE_PROJECT_REF>.functions.supabase.co/gumroad-webhook-handler`
  - 체크아웃/리다이렉트에 `user_id=<supabase auth user id>` 쿼리 파라미터를 포함하면, 핑 payload의 `url_params.user_id`로 전달되어 즉시 후원자 플래그가 적용됩니다.
  - 이메일로 매핑도 지원: `email`이 포함된 경우 `get_user_id_by_email` RPC로 사용자 ID를 조회합니다.
