export type StonePlayer = "black" | "white";

export interface SampleMeta {
  source?: string;
  gameId?: string;
  moveIndex?: number;
  totalMoves?: number;
  tags?: string[];
  result?: -1 | 0 | 1;
  extra?: Record<string, unknown>;
}

export interface TrainingSample {
  state: (StonePlayer | null)[][];
  player: StonePlayer;
  mcts_policy: number[];
  teacher_policy: number[];
  teacher_value: number;
  final_value: -1 | 0 | 1;
  meta?: SampleMeta;
}
