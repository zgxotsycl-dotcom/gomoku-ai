import type { Player } from './ai';

export type Board = (Player | null)[][];

function inside(n: number, r: number, c: number) {
  return r >= 0 && c >= 0 && r < n && c < n;
}

function firstEmptyAround(board: Board, r0: number, c0: number, rings = 2): [number, number] {
  const n = board.length;
  for (let rad = 1; rad <= rings; rad++) {
    for (let dr = -rad; dr <= rad; dr++) {
      for (let dc = -rad; dc <= rad; dc++) {
        const r = r0 + dr, c = c0 + dc;
        if (!inside(n, r, c)) continue;
        if (board[r][c] === null) return [r, c];
      }
    }
  }
  // fallback: first global empty
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (board[r][c] === null) return [r, c];
  return [-1, -1];
}

// Minimal Swap2 opener: propose BWB near center, then responder chooses White and places one more White.
// This is a lightweight approximation intended to start games from a Swap2-legal position in self-play/arena.
export function generateSwap2Opening(board: Board): { board: Board; toMove: Player } {
  const n = board.length;
  const mid = Math.floor(n / 2);
  // Start from copy to avoid mutating caller unexpectedly
  const b = board.map(row => row.slice()) as Board;

  // Step 1: proposer places B at center
  if (b[mid][mid] == null) b[mid][mid] = 'black';
  // Place W adjacent
  const w1 = firstEmptyAround(b, mid, mid, 1);
  if (w1[0] !== -1) b[w1[0]][w1[1]] = 'white';
  // Place second B near center as well
  const b2 = firstEmptyAround(b, mid, mid, 1);
  if (b2[0] !== -1) b[b2[0]][b2[1]] = 'black';

  // Step 2: responder chooses to play White and places an extra White stone (common simple option)
  const w2 = firstEmptyAround(b, mid, mid, 1);
  if (w2[0] !== -1) b[w2[0]][w2[1]] = 'white';

  // After this choice, last stone was White, so next is Black to move
  return { board: b, toMove: 'black' };
}

