import { parentPort, workerData } from 'node:worker_threads';
import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';
import { pathToFileURL } from 'node:url';
import {
  findBestMoveNN,
  checkWin,
  getOpponent,
  type Player,
  type PolicyData,
} from './ai';
import { performSwap2Negotiation } from './swap2_negotiation';

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

/* ======================
 * 타입 선언
 * ====================== */
type Move = [number, number];
type Board = (Player | null)[][];
type GameResult = -1 | 0 | 1;

interface TrainingStep {
  state: Board;
  player: Player;
  mcts_policy: number[];     // 길이 BOARD_SIZE*BOARD_SIZE
  teacher_policy: number[];  // 모델 예측 정책
  teacher_value: number;     // 모델 예측 가치(스칼라)
}

interface TrainingSample extends TrainingStep {
  final_value: GameResult;
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
  return pathToFileURL(p).href;
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
      const { board: opened, toMove, swapColors } = await performSwap2Negotiation(board, modelWhite); // second player initially is White
      board = opened;
      player = toMove;
      if (swapColors) {
        // Second player chose to be Black (or P1 chose White in opt3) -> swap model roles
        models = { black: modelWhite, white: modelBlack };
      }
    }
  } catch {}
  const history: TrainingStep[] = [];

  const maxMoves = BOARD_SIZE * BOARD_SIZE;

  for (let moveCount = 0; moveCount < maxMoves; moveCount++) {
    const currentModel = models[player];

    // 1) 교사(현재 모델) 정책/가치 예측
    const teacher = getTeacherPrediction(currentModel, board, player);

    // 2) MCTS로 최적 수/정책 도출
    // Dynamic think time by game phase to diversify self-play strength
    let thinkTime = MCTS_THINK_TIME_MS;
    if (moveCount <= 6) thinkTime = Math.max(1000, Math.floor(MCTS_THINK_TIME_MS * 0.8));
    else if (moveCount <= 30) thinkTime = Math.max(1200, Math.floor(MCTS_THINK_TIME_MS * 1.2));
    else thinkTime = Math.max(800, Math.floor(MCTS_THINK_TIME_MS * 1.0));

    const { bestMove, policy: mctsPolicy } = await findBestMoveNN(
      currentModel,
      board,
      player,
      thinkTime
    );

    if (!bestMove || bestMove[0] === -1 || bestMove[1] === -1) {
      // 둘 곳이 없거나 에러 시 종료(무승부 처리)
      break;
    }

    // 3) MCTS 방문수 기반 타깃 정책 생성
    const policyTarget = new Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    let totalVisits = 0;
    for (const p of mctsPolicy as PolicyData[]) totalVisits += p.visits;

    if (totalVisits > 0) {
      for (const p of mctsPolicy as PolicyData[]) {
        const moveIndex = p.move[0] * BOARD_SIZE + p.move[1];
        policyTarget[moveIndex] = p.visits / totalVisits;
      }
    }

    // 4) 기록(상태, 교사, MCTS 타깃)
    history.push({
      state: cloneBoard(board),
      player,
      mcts_policy: policyTarget,
      teacher_policy: teacher.teacher_policy,
      teacher_value: teacher.teacher_value,
    });

    // 5) 수 선택(초반 탐험: 확률 샘플링, 이후: 최빈/최대방문)
    let chosen: Move;
    if (moveCount < EXPLORATION_MOVES && (mctsPolicy as PolicyData[]).length > 1) {
      // 방문 비율로 직접 샘플링(텐서 생성 X)
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

    // 6) 착수 유효성 방어
    const [r, c] = chosen;
    if (r < 0 || c < 0 || r >= BOARD_SIZE || c >= BOARD_SIZE || board[r][c] !== null) {
      console.warn(`[Worker ${cfg.workerId ?? '?'}] Illegal move attempted:`, chosen);
      break; // 무승부 처리
    }

    // 7) 보드 업데이트
    board[r][c] = player;

    // 8) 승리 체크
    if (checkWin(board, player, chosen)) {
      const winner = player;
      const trainingSamples: TrainingSample[] = history.map((h) => ({
        ...h,
        final_value: h.player === winner ? 1 : -1,
      }));
      parentPort?.postMessage({ trainingSamples });
      return;
    }

    // 9) 차례 넘기기
    player = getOpponent(player);
  }

  // 가득 찼거나 유효 수 없음 → 무승부
  const trainingSamples: TrainingSample[] = history.map((h) => ({
    ...h,
    final_value: 0,
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
