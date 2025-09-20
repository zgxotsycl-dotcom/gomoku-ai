import * as fs from 'fs/promises';
import * as path from 'path';
import { getPossibleMoves, getOpponent, findThreats } from './ai'; // Assuming findThreats is in ai.ts
import type { Player, Move } from './ai';

const REPLAY_BUFFER_DIR = path.resolve(__dirname, '../replay_buffer');
const OUTPUT_FILE = path.resolve(__dirname, '../subgoal_training_data.jsonl');

interface GameSample {
    state: (Player | null)[][];
    mcts_policy: number[];
    final_value: number;
    player: Player;
}

interface SubgoalSample {
    state: (Player | null)[][];
    subgoal: 'create_open_four' | 'block_open_four';
    action_policy: number[]; // One-hot encoded target move
}

async function main() {
    console.log('--- Starting Subgoal Dataset Generation ---');
    const startTime = Date.now();
    let subgoalSamples: SubgoalSample[] = [];

    const files = await fs.readdir(REPLAY_BUFFER_DIR);
    console.log(`Found ${files.length} game data files in ${REPLAY_BUFFER_DIR}.`);

    for (const file of files) {
        if (!file.endsWith('.jsonl')) continue;

        const fileContent = await fs.readFile(path.join(REPLAY_BUFFER_DIR, file), 'utf-8');
        const gameSamples: GameSample[] = fileContent.trim().split('\n').map(line => JSON.parse(line));

        for (let i = 0; i < gameSamples.length - 1; i++) {
            const currentSample = gameSamples[i];
            const nextSample = gameSamples[i+1];
            const board = currentSample.state;
            // Enforce unified 15x15 dataset
            if (!board || board.length !== 15) continue;
            const player = currentSample.player;
            const opponent = getOpponent(player);
            
            // Find the actual move made
            let actualMove: [number, number] | null = null;
            const N = board.length;
            for(let r=0; r<N; r++){
                for(let c=0; c<N; c++){
                    if(board[r][c] !== nextSample.state[r][c]){
                        actualMove = [r,c];
                        break;
                    }
                }
                if(actualMove) break;
            }
            if(!actualMove) continue;

            // --- Mission 1: Create Open Four ---
            const createOpenFourMoves = findThreats(board, player, 'open-four');
            if (createOpenFourMoves.some((m: Move) => m[0] === actualMove![0] && m[1] === actualMove![1])) {
                const policy = new Array(N * N).fill(0);
                policy[actualMove[0] * N + actualMove[1]] = 1;
                subgoalSamples.push({ state: board, subgoal: 'create_open_four', action_policy: policy });
            }

            // --- Mission 2: Block Opponent's Open Four ---
            const blockOpenFourMoves = findThreats(board, opponent, 'open-four');
            if (blockOpenFourMoves.some((m: Move) => m[0] === actualMove![0] && m[1] === actualMove![1])) {
                const policy = new Array(N * N).fill(0);
                policy[actualMove[0] * N + actualMove[1]] = 1;
                subgoalSamples.push({ state: board, subgoal: 'block_open_four', action_policy: policy });
            }
        }
    }

    console.log('\n--- Dataset Generation Complete ---');
    console.log(`Found ${subgoalSamples.length} subgoal samples.`);

    if (subgoalSamples.length > 0) {
        const data = subgoalSamples.map(s => JSON.stringify(s)).join('\n') + '\n';
        await fs.writeFile(OUTPUT_FILE, data);
        console.log(`Subgoal dataset saved to ${OUTPUT_FILE}`);
    }

    const endTime = Date.now();
    console.log(`Finished in ${(endTime - startTime) / 1000} seconds.`);
}

main().catch(console.error);
