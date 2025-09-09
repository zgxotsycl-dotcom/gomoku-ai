const fs = require('fs');

// --- AI Logic (Copied and adapted for Node.js) ---
const BOARD_SIZE = 19;

const checkWin = (board, player, row, col) => {
  const directions = [{ x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }, { x: 1, y: -1 }];
  for (const dir of directions) {
    let count = 1;
    for (let i = 1; i < 5; i++) {
      const newRow = row + i * dir.y; const newCol = col + i * dir.x;
      if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE || board[newRow][newCol] !== player) break;
      count++;
    }
    for (let i = 1; i < 5; i++) {
      const newRow = row - i * dir.y; const newCol = col - i * dir.x;
      if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE || board[newRow][newCol] !== player) break;
      count++;
    }
    if (count >= 5) return true;
  }
  return false;
};

function getPossibleMoves(board) {
    const moves = new Set();
    const radius = 2;
    let hasStones = false;
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] !== null) {
                hasStones = true;
                for (let i = -radius; i <= radius; i++) {
                    for (let j = -radius; j <= radius; j++) {
                        const newR = r + i;
                        const newC = c + j;
                        if (newR >= 0 && newR < BOARD_SIZE && newC >= 0 && newC < BOARD_SIZE && board[newR][newC] === null) {
                            moves.add(`${newR},${newC}`);
                        }
                    }
                }
            }
        }
    }
    if (!hasStones) {
        return [[Math.floor(BOARD_SIZE / 2), Math.floor(BOARD_SIZE / 2)]];
    }
    return Array.from(moves).map(move => {
        const [r, c] = move.split(',').map(Number);
        return [r, c];
    });
}

const HEURISTIC_SCORE = {
    FIVE: 100000, OPEN_FOUR: 10000, CLOSED_FOUR: 1000,
    OPEN_THREE: 500, CLOSED_THREE: 10, OPEN_TWO: 5, CLOSED_TWO: 1,
};

function evaluateLine(line, player) {
    const opponent = player === 'black' ? 'white' : 'black';
    let score = 0;
    const playerCount = line.filter(cell => cell === player).length;
    const emptyCount = line.filter(cell => cell === null).length;
    if (playerCount === 5) return HEURISTIC_SCORE.FIVE;
    if (playerCount === 4 && emptyCount === 1) score += HEURISTIC_SCORE.OPEN_FOUR;
    if (playerCount === 3 && emptyCount === 2) score += HEURISTIC_SCORE.OPEN_THREE;
    if (playerCount === 2 && emptyCount === 3) score += HEURISTIC_SCORE.OPEN_TWO;
    const opponentCount = line.filter(cell => cell === opponent).length;
    if (opponentCount === 5) return -HEURISTIC_SCORE.FIVE;
    if (opponentCount === 4 && emptyCount === 1) score -= HEURISTIC_SCORE.OPEN_FOUR * 1.5;
    if (opponentCount === 3 && emptyCount === 2) score -= HEURISTIC_SCORE.OPEN_THREE * 1.5;
    return score;
}

function evaluateBoard(board, player) {
    let totalScore = 0;
    for (let r = 0; r < BOARD_SIZE; r++) { for (let c = 0; c <= BOARD_SIZE - 5; c++) { totalScore += evaluateLine(board[r].slice(c, c + 5), player); } }
    for (let c = 0; c < BOARD_SIZE; c++) { for (let r = 0; r <= BOARD_SIZE - 5; r++) { const line = []; for (let i = 0; i < 5; i++) line.push(board[r + i][c]); totalScore += evaluateLine(line, player); } }
    for (let r = 0; r <= BOARD_SIZE - 5; r++) { for (let c = 0; c <= BOARD_SIZE - 5; c++) { const line = []; for (let i = 0; i < 5; i++) line.push(board[r + i][c + i]); totalScore += evaluateLine(line, player); } }
    for (let r = 0; r <= BOARD_SIZE - 5; r++) { for (let c = 4; c < BOARD_SIZE; c++) { const line = []; for (let i = 0; i < 5; i++) line.push(board[r + i][c - i]); totalScore += evaluateLine(line, player); } }
    return totalScore;
}

function minimax(board, depth, alpha, beta, maximizingPlayer, aiPlayer) {
    if (depth === 0) { return evaluateBoard(board, aiPlayer); }
    const possibleMoves = getPossibleMoves(board);
    const humanPlayer = aiPlayer === 'black' ? 'white' : 'black';
    if (maximizingPlayer) {
        let maxEval = -Infinity;
        for (const [r, c] of possibleMoves) {
            const newBoard = board.map(row => [...row]);
            newBoard[r][c] = aiPlayer;
            const evaluation = minimax(newBoard, depth - 1, alpha, beta, false, aiPlayer);
            maxEval = Math.max(maxEval, evaluation);
            alpha = Math.max(alpha, evaluation);
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (const [r, c] of possibleMoves) {
            const newBoard = board.map(row => [...row]);
            newBoard[r][c] = humanPlayer;
            const evaluation = minimax(newBoard, depth - 1, alpha, beta, true, aiPlayer);
            minEval = Math.min(minEval, evaluation);
            beta = Math.min(beta, evaluation);
            if (beta <= alpha) break;
        }
        return minEval;
    }
}

function findBestMoveForTraining(board, player) {
    const opponent = player === 'black' ? 'white' : 'black';
    const possibleMoves = getPossibleMoves(board);
    if (possibleMoves.length === 0) return [-1, -1];
    for (const [r, c] of possibleMoves) { const newBoard = board.map(row => [...row]); newBoard[r][c] = player; if (checkWin(newBoard, player, r, c)) return [r, c]; }
    for (const [r, c] of possibleMoves) { const newBoard = board.map(row => [...row]); newBoard[r][c] = opponent; if (checkWin(newBoard, opponent, r, c)) return [r, c]; }
    let bestVal = -Infinity;
    let bestMove = possibleMoves[0] || [-1, -1];
    const searchDepth = 2; // Reduced depth for faster training
    for (const [r, c] of possibleMoves) {
        const newBoard = board.map(row => [...row]);
        newBoard[r][c] = player;
        const moveVal = minimax(newBoard, searchDepth, -Infinity, Infinity, false, player);
        if (moveVal > bestVal) {
            bestMove = [r, c];
            bestVal = moveVal;
        }
    }
    return bestMove;
}

// --- Training Logic ---
const boardToString = (board) => board.map(row => row.map(cell => cell ? (cell === 'black' ? 'b' : 'w') : '-').join('')).join('|');
const openingBook = new Map();
let positionsCalculated = 0;

function generateOpenings(board, currentPlayer, moveCount, maxMoveCount) {
    if (moveCount >= maxMoveCount) { return; }
    const boardHash = boardToString(board);
    if (openingBook.has(boardHash)) { return; }
    const bestMove = findBestMoveForTraining(board, currentPlayer);
    openingBook.set(boardHash, { best_move: bestMove, move_count: moveCount });
    positionsCalculated++;
    if (positionsCalculated % 100 === 0) {
        console.log(`Calculated ${positionsCalculated} positions...`);
    }
    const possibleMoves = getPossibleMoves(board);
    for (const [r, c] of possibleMoves) {
        const newBoard = board.map(row => [...row]);
        newBoard[r][c] = currentPlayer;
        if (checkWin(newBoard, currentPlayer, r, c)) { continue; }
        generateOpenings(newBoard, currentPlayer === 'black' ? 'white' : 'black', moveCount + 1, maxMoveCount);
    }
}

async function main() {
    console.log("Starting Opening Book generation...");
    const startTime = Date.now();
    const initialBoard = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    const maxDepth = 3; // Generate for the first 3 moves
    console.log(`Generating openings up to ${maxDepth} moves deep...`);
    generateOpenings(initialBoard, 'black', 0, maxDepth);
    console.log("...Generation complete.");
    console.log(`Total positions in opening book: ${openingBook.size}`);
    const outputData = Array.from(openingBook.entries()).map(([hash, data]) => ({
        board_hash: hash,
        best_move: data.best_move,
        move_count: data.move_count
    }));
    try {
        console.log(`Attempting to write ${outputData.length} entries to file...`);
        const jsonString = JSON.stringify(outputData, null, 2);
        console.log(`JSON string length: ${jsonString.length}`);
        fs.writeFileSync('opening_book.json', jsonString);
        console.log("fs.writeFileSync finished.");
    } catch (e) {
        console.error("!!! FAILED TO WRITE FILE !!!");
        console.error(e);
    }
    const endTime = Date.now();
    console.log(`
Finished in ${(endTime - startTime) / 1000} seconds.`);
    console.log("Training data saved to opening_book.json");
}

main();