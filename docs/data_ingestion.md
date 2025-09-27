# Professional Game Data Ingestion

This guide describes how to convert human/professional Gomoku games into the replay buffer format that the training pipeline consumes.

## Supported input formats
- `.json` / `.jsonl`: objects with `moves` and optional `winner`, for example:
  ```json
  {
    "board_size": 15,
    "first_player": "black",
    "winner": "black",
    "moves": [ [7,7, "black"], [7,8, "white"], [8,8, "black"] ]
  }
  ```
- `.csv`: rows with `moves` (semicolon-separated `row col player?` tokens), optional `winner`, `board_size`.

All coordinates are zero-indexed. If `player` is omitted the importer alternates starting with `PRO_FIRST_PLAYER` (default `black`).

## Running the importer
Build the project once (`npm run build`), then run:
```bash
node dist/data/ingest_pro_games.js --input data/pro_games --output replay_buffer
```
Environment variables:
- `PRO_GAME_DIR`: default `data/pro_games`
- `PRO_GAME_OUTPUT`: default `replay_buffer`
- `BOARD_SIZE`: override board size if absent in files
- `PRO_FIRST_PLAYER`: `black` or `white`

Each run creates a JSONL file in the output directory. Entries already match the training sample schema (state, policy, value labels) so the distillation step can consume them immediately.

## Workflow tips
- Place raw pro games under `data/pro_games/` (subdirectories allowed).
- After ingestion, the generated JSONL files sit alongside self-play data inside `replay_buffer/`. You can keep them in a separate subfolder and set `PRO_GAME_OUTPUT` accordingly if you prefer manual curation.
- Consider tagging the filename when running the importer, e.g. `PRO_GAME_OUTPUT=replay_buffer PRO_GAME_TAG=pro` (set manually by renaming the resulting file) to keep track of data provenance.
- Use `MAX_FILES_PER_EPOCH` or `PRIORITIZE_RECENT_FILES=false` (distillation env vars) to balance self-play vs human data.


## Gomocup PSQ Import
- Place `.psq` files under a directory (recursively).
- Run `npm run build` followed by `npm run data:ingest:psq -- --input <psq_dir> --output replay_buffer`.
- Important env vars: `PSQ_BOARD_SIZE` (default 15), `PSQ_PLAYER1_IS_BLACK` (default true), `PSQ_OUTPUT_DIR`.
- Assumes Player1가 첫 수(흑)를 두는 규칙입니다. 필요 시 `PSQ_PLAYER1_IS_BLACK=false`로 조정하세요.


### PSQ Batch Automation
- Organise yearly Gomocup folders under `PSQ_BATCH_ROOT` (default `data/psq_batches`).
- Run `npm run build && npm run data:ingest:psq:batch`; the script ingests each subdirectory once and records progress in `.psq_processed.json`.
- Configure environment variables: `PSQ_BATCH_OUTPUT`, `PSQ_BATCH_STATE`, `PSQ_BATCH_ROOT`.
