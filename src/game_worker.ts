import { parentPort, workerData } from 'node:worker_threads';
import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';
import * as path from 'path';
import {
  findBestMoveNN,
  checkWin,
  getOpponent,
  type Player,
  type PolicyData,
} from './ai';
import { performSwap2Negotiation } from './swap2_negotiation';
import type { TrainingSample, SampleMeta } from './types/training';

/* ======================
 * 설정값 (workerData로 덮어쓰기 가능)
 * ====================== */
const BOARD_SIZE: number = (workerData?.boardSize as number) ?? Number(process.env.BOARD_SIZE || 15);
// Time-based MCTS think time (ms)
const MCTS_THINK_TIME_MS: number =
  (workerData?.mctsSimLimit as number) ?? 1600;
const EXPLORATION_MOVES: number =
  (workerData?.explorationMoves as number) ?? 15; // 초기 탐험 구간(샘플링)
const EPS = 1e-8;
interface ThinkScheduleEntry {
  move: number;
  ms: number;
}

function parseThinkSchedule(raw?: string): ThinkScheduleEntry[] | null {
  if (!raw) return null;
  const entries: ThinkScheduleEntry[] = [];
  for (const part of raw.split(/[;,]/)) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const [moveStr, valueStr] = trimmed.split(':');
    const move = Number(moveStr?.trim());
    const ms = Number(valueStr?.trim());
    if (Number.isFinite(move) && Number.isFinite(ms)) {
      entries.push({ move: Math.max(0, Math.floor(move)), ms: Math.max(200, Math.floor(ms)) });
    }
  }
  if (entries.length === 0) return null;
  entries.sort((a, b) => a.move - b.move);
  return entries;
}

const THINK_TIME_SCHEDULE = parseThinkSchedule(process.env.MCTS_THINK_TIME_SCHEDULE);
const THINK_TIME_JITTER = Number(process.env.MCTS_THINK_TIME_JITTER || 0);

function defaultThinkTime(moveIndex: number): number {
  if (moveIndex <= 6) return Math.max(1000, Math.floor(MCTS_THINK_TIME_MS * 0.8));
  if (moveIndex <= 30) return Math.max(1200, Math.floor(MCTS_THINK_TIME_MS * 1.2));
  return Math.max(800, Math.floor(MCTS_THINK_TIME_MS));
}

function resolveThinkTime(moveIndex: number): number {
  let base = defaultThinkTime(moveIndex);
  if (THINK_TIME_SCHEDULE) {
    for (const entry of THINK_TIME_SCHEDULE) {
      if (moveIndex >= entry.move) base = entry.ms;
      else break;
    }
  }
  if (Number.isFinite(THINK_TIME_JITTER) && THINK_TIME_JITTER > 0) {
    const amplitude = Math.abs(THINK_TIME_JITTER);
    const delta = base * amplitude * (Math.random() * 2 - 1);
    base = Math.max(200, base + delta);
  }
  return Math.max(200, Math.floor(base));
}

/* ======================
 * 타입 선언
 * ====================== */
type Move = [number, number];
type Board = (Player | null)[][];
type GameResult = -1 | 0 | 1;

interface TrainingStep {
  state: Board;
  player: Player;
  mcts_policy: number[];
  teacher_policy: number[];
  teacher_value: number;
  meta?: SampleMeta;
}

interface WorkerConfig {
  workerId?: string | number;
  prodModelPath: string;
  opponentModelPath?: string | null;
  autostart?: boolean;
  boardSize?: number;
  mctsSimLimit?: number;
  explorationMoves?: number;
}

/* ======================
 * 전역 상태(모델 캐시 & 실행 가드)
 * ====================== */
const cfg = workerData as WorkerConfig;
let isRunning = false;

const modelCache: {
  black: TFT.LayersModel | null;
  white: TFT.LayersModel | null;
  paths: { black?: string; white?: string };
} = { black: null, white: null, paths: {} };

/* ======================
 * 유틸리티
 * ====================== */

/** board를 [1, H, W, C=3] 텐서로 변환(플레이어, 상대, 차례 색상) */
function boardToInputTensor(board: Board, player: Player): TFT.Tensor4D {
  return tf.tidy(() => {
    const size = BOARD_SIZE;
    const channels = 3; // [player, opponent, side]
    const data = new Float32Array(size * size * channels);
    const opponent = player === 'black' ? 'white' : 'black';
    const sideVal = player === 'black' ? 1 : 0;

    let idx = 0;
    for (let r = 0; r < size; r++) {
      const row = board[r];
      for (let c = 0; c < size; c++) {
        const cell = row[c];
        // ch 0: 현재 플레이어 돌
        data[idx] = cell === player ? 1 : 0;
        // ch 1: 상대 돌
        data[idx + 1] = cell === opponent ? 1 : 0;
        // ch 2: 현재 플레이어 색상(black=1, white=0)
        data[idx + 2] = sideVal;
        idx += channels;
      }
    }
    return tf.tensor4d(data, [1, size, size, channels]);
  });
}

/** Windows 경로 안전히 파일 URL로 변환 */
function toFileURL(p: string): string {
  const resolved = path.resolve(p).replace(/\\/g, '/');
  if (/^[a-zA-Z]:\//.test(resolved)) {
    return `file://${resolved}`;
  }
  const prefixed = resolved.startsWith('/') ? resolved : `/${resolved}`;
  return `file://${prefixed}`;
}

/** 레이어 모델 로드 + 워밍업(첫 predict에서 그래프 컴파일되는 비용 선지불) */
async function loadModel(modelPath: string): Promise<TFT.LayersModel> {
  const url = toFileURL(modelPath);
  console.log(`[Worker ${cfg.workerId ?? '?'}] Loading model: ${url}`);
  const m = await tf.loadLayersModel(url);
  // Warm-up 1x15x15x3
  tf.tidy(() => {
    const dummy = tf.zeros([1, BOARD_SIZE, BOARD_SIZE, 3]);
    const out = m.predict(dummy);
    // out 텐서는 tidy가 해제
    void out;
  });
  return m;
}

/** 모델 캐시 보장 */
async function ensureModels(): Promise<{ black: TFT.LayersModel; white: TFT.LayersModel }> {
  const reqBlack = cfg.prodModelPath;
  const reqWhite = cfg.opponentModelPath ?? cfg.prodModelPath;

  // 필요 시 로드/리로드
  if (!modelCache.black || modelCache.paths.black !== reqBlack) {
    if (modelCache.black) modelCache.black.dispose();
    modelCache.black = await loadModel(reqBlack);
    modelCache.paths.black = reqBlack;
  }
  if (!modelCache.white || modelCache.paths.white !== reqWhite) {
    if (modelCache.white && modelCache.white !== modelCache.black) modelCache.white.dispose();
    modelCache.white = reqWhite === reqBlack ? modelCache.black! : await loadModel(reqWhite);
    modelCache.paths.white = reqWhite;
  }
  return { black: modelCache.black!, white: modelCache.white! };
}

/** 모델 출력 검증 및 추출 */
function getTeacherPrediction(
  model: TFT.LayersModel,
  board: Board,
  player: Player
): { teacher_policy: number[]; teacher_value: number } {
  return tf.tidy(() => {
    const input = boardToInputTensor(board, player);
    const out = model.predict(input);
    if (!Array.isArray(out) || out.length < 2) {
      throw new Error(
        'Model must output [policyTensor, valueTensor]. Check your model outputs.'
      );
    }
    const policyTensor = out[0];
    const valueTensor = out[1];
    const teacher_policy = Array.from(policyTensor.dataSync() as Float32Array);
    const teacher_value = (valueTensor.dataSync() as Float32Array)[0];

    // NaN/Inf 방어
    for (let i = 0; i < teacher_policy.length; i++) {
      const v = teacher_policy[i];
      if (!isFinite(v)) teacher_policy[i] = 0;
    }
    if (!isFinite(teacher_value)) {
      console.warn('[Teacher] value was NaN/Inf. Clamping to 0.');
      return { teacher_policy, teacher_value: 0 };
    }
    return { teacher_policy, teacher_value };
  });
}

/** 확률 벡터 정규화(합=1), 모두 0이면 균등분포 */
function normalizeProbs(arr: number[], eps = EPS): number[] {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += Math.max(arr[i], 0);
  if (!isFinite(s) || s <= 0) {
    const u = 1 / arr.length;
    return arr.map(() => u);
  }
  s = Math.max(s, eps);
  return arr.map((x) => Math.max(x, 0) / s);
}

/** 확률에 따른 CPU 샘플링(의존성/텐서 생성 없이 가볍게) */
function sampleIndex(probs: number[]): number {
  const p = normalizeProbs(probs);
  const r = Math.random(); // 필요시 seedable RNG로 치환
  let acc = 0;
  for (let i = 0; i < p.length; i++) {
    acc += p[i];
    if (r <= acc) return i;
  }
  return p.length - 1; // 수치오차 방어
}

/** 보드 깊은 복사(JSON 대신 slice로 2~4배 빠름) */
function cloneBoard(b: Board): Board {
  const out: Board = new Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b[i].slice();
  return out;
}

/* ======================
 * Self-Play 본체
 * ====================== */
async function runSelfPlayGame(): Promise<void> {
  const { black: modelBlack, white: modelWhite } = await ensureModels();
  let models: Record<Player, TFT.LayersModel> = { black: modelBlack, white: modelWhite };

  let board: Board = Array.from({ length: BOARD_SIZE }, () =>
    Array<Player | null>(BOARD_SIZE).fill(null)
  );
  let player: Player = 'black';
  // Apply Swap2 negotiation if enabled
  try {
    const rule = (process.env.OPENING_RULE || 'swap2').toLowerCase();
    if (rule === 'swap2') {
      const { board: opened, toMove, swapColors } = await performSwap2Negotiation(board, modelWhite);
      board = opened;
      player = toMove;
      if (swapColors) {
        models = { black: modelWhite, white: modelBlack };
      }
    }
  } catch {}
  const history: TrainingStep[] = [];
  const gameId = `${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
  const baseTags = ['self_play'];

  const maxMoves = BOARD_SIZE * BOARD_SIZE;

  for (let moveCount = 0; moveCount < maxMoves; moveCount++) {
    const currentModel = models[player];

    // 1) Teacher prediction
    const teacher = getTeacherPrediction(currentModel, board, player);

    // 2) MCTS-guided search with configurable think time
    const thinkTime = resolveThinkTime(moveCount);
    const { bestMove, policy: mctsPolicy } = await findBestMoveNN(
      currentModel,
      board,
      player,
      thinkTime
    );

    if (!bestMove || bestMove[0] === -1 || bestMove[1] === -1) {
      break;
    }

    // 3) Convert visit counts to policy distribution
    const policyTarget = new Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    let totalVisits = 0;
    for (const p of mctsPolicy as PolicyData[]) totalVisits += p.visits;
    if (totalVisits > 0) {
      for (const p of mctsPolicy as PolicyData[]) {
        const moveIndex = p.move[0] * BOARD_SIZE + p.move[1];
        policyTarget[moveIndex] = p.visits / totalVisits;
      }
    }

    const isExplorationPhase = moveCount < EXPLORATION_MOVES;
    const tags = [...baseTags, `player:${player}`];
    if (isExplorationPhase) tags.push('exploration');

    // 4) Record training snapshot with metadata
    history.push({
      state: cloneBoard(board),
      player,
      mcts_policy: policyTarget,
      teacher_policy: teacher.teacher_policy,
      teacher_value: teacher.teacher_value,
      meta: {
        source: 'self_play',
        gameId,
        moveIndex: moveCount,
        tags,
        extra: { thinkTime, exploration: isExplorationPhase },
      },
    });

    // 5) Exploration vs exploitation move selection
    let chosen: Move;
    if (isExplorationPhase && (mctsPolicy as PolicyData[]).length > 1) {
      const moves = (mctsPolicy as PolicyData[]).map((p) => p.move) as Move[];
      const probs =
        totalVisits > 0
          ? (mctsPolicy as PolicyData[]).map((p) => p.visits / totalVisits)
          : moves.map(() => 1 / moves.length);
      const idx = sampleIndex(probs);
      chosen = moves[idx];
    } else {
      chosen = bestMove as Move;
    }

    // 6) Validate move
    const [r, c] = chosen;
    if (r < 0 || c < 0 || r >= BOARD_SIZE || c >= BOARD_SIZE || board[r][c] !== null) {
      console.warn(`[Worker ${cfg.workerId ?? '?'}] Illegal move attempted:`, chosen);
      break;
    }

    // 7) Apply move
    board[r][c] = player;

    // 8) Check win
    if (checkWin(board, player, chosen)) {
      const winner = player;
      const totalMoves = history.length;
      const blackResult = winner === 'black' ? 1 : -1;
      const resultTag = `winner:${winner}`;
      const trainingSamples: TrainingSample[] = history.map((h, idx) => ({
        state: h.state,
        player: h.player,
        mcts_policy: h.mcts_policy,
        teacher_policy: h.teacher_policy,
        teacher_value: h.teacher_value,
        final_value: h.player === winner ? 1 : -1,
        meta: {
          ...(h.meta ?? {}),
          source: h.meta?.source ?? 'self_play',
          gameId,
          moveIndex: idx,
          totalMoves,
          result: blackResult,
          tags: Array.from(new Set([...(h.meta?.tags ?? []), resultTag])),
          extra: { ...(h.meta?.extra ?? {}) },
        },
      }));
      parentPort?.postMessage({ trainingSamples });
      return;
    }

    // 9) Switch player
    player = getOpponent(player);
  }

  // Draw or exhausted moves
  const drawTotalMoves = history.length;
  const trainingSamples: TrainingSample[] = history.map((h, idx) => ({
    state: h.state,
    player: h.player,
    mcts_policy: h.mcts_policy,
    teacher_policy: h.teacher_policy,
    teacher_value: h.teacher_value,
    final_value: 0,
    meta: {
      ...(h.meta ?? {}),
      source: h.meta?.source ?? 'self_play',
      gameId,
      moveIndex: idx,
      totalMoves: drawTotalMoves,
      result: 0,
      tags: Array.from(new Set([...(h.meta?.tags ?? []), 'result:draw'])),
      extra: { ...(h.meta?.extra ?? {}) },
    },
  }));
  parentPort?.postMessage({ trainingSamples });
}
/* ======================
 * 실행/메시지 처리
 * ====================== */
async function startGame() {
  if (isRunning) {
    console.warn(`[Worker ${cfg.workerId ?? '?'}] Game is already running. Ignoring new request.`);
    return;
  }
  isRunning = true;
  try {
    await runSelfPlayGame();
  } catch (e) {
    console.error(`[Worker ${cfg.workerId ?? '?'}] Error during self-play:`, e);
  } finally {
    isRunning = false;
  }
}

parentPort?.on('message', async (msg) => {
  if (msg === 'start_new_game') {
    await startGame();
  } else if (msg?.type === 'reload_models') {
    // 동적 리로드 지원(선택)
    try {
      if (msg.prodModelPath) cfg.prodModelPath = msg.prodModelPath;
      if ('opponentModelPath' in msg) cfg.opponentModelPath = msg.opponentModelPath;
      // 다음 startGame에서 자동 반영(ensureModels가 캐시 갱신)
      parentPort?.postMessage({ reloaded: true });
    } catch (e) {
      parentPort?.postMessage({ reloaded: false, error: String(e) });
    }
  }
});

// 기본은 자동 시작(기존 동작과 호환). 중복 실행은 isRunning으로 방지.
if (cfg.autostart ?? true) {
  void startGame();
}

























