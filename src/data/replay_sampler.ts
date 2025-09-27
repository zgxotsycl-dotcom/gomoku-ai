import type { TrainingSample } from "../types/training";

interface BalanceConfig {
  enabled: boolean;
  ratios: [number, number, number]; // [loss, draw, win]
  tolerance: number;
}

interface FilterConfig {
  minMoveIndex?: number;
  maxMoveIndex?: number;
  minTotalMoves?: number;
  maxTotalMoves?: number;
  includeSources?: Set<string>;
  excludeSources?: Set<string>;
  minPolicyEntropy?: number;
  maxPolicyEntropy?: number;
}

export interface ReplaySamplerStats {
  seen: number;
  accepted: number;
  rejected: number;
  filtered: { reason: string; count: number }[];
  finalValueCounts: { "-1": number; "0": number; "1": number };
}

type FilterReason = "source" | "move" | "policy" | "balance";

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (!value) return fallback;
  const v = value.toLowerCase();
  if (["1", "true", "yes", "on"].includes(v)) return true;
  if (["0", "false", "no", "off"].includes(v)) return false;
  return fallback;
}

function parseRatios(input: string | undefined): [number, number, number] {
  if (!input) return [1, 1, 1];
  const parts = input.split(/[,:]/).map((p) => Number(p.trim())).filter((n) => Number.isFinite(n) && n >= 0);
  if (parts.length >= 3) return [parts[0], parts[1], parts[2]];
  if (parts.length === 1) return [parts[0], parts[0], parts[0]];
  return [1, 1, 1];
}

function normalizeRatios(r: [number, number, number]): [number, number, number] {
  const sum = r[0] + r[1] + r[2];
  if (sum <= 0) return [1 / 3, 1 / 3, 1 / 3];
  return [r[0] / sum, r[1] / sum, r[2] / sum];
}

function shannonEntropy(dist: number[]): number {
  let entropy = 0;
  for (const p of dist) {
    if (p > 0) entropy -= p * Math.log(p);
  }
  return entropy;
}

class Counter {
  private map = new Map<FilterReason, number>();

  inc(reason: FilterReason): void {
    this.map.set(reason, (this.map.get(reason) ?? 0) + 1);
  }

  toArray(): { reason: string; count: number }[] {
    return Array.from(this.map.entries()).map(([reason, count]) => ({ reason, count }));
  }
}

class ReplaySampler {
  private balanceCounts: [number, number, number] = [0, 0, 0];
  private totalSeen = 0;
  private totalAccepted = 0;
  private rejects = new Counter();
  private readonly balance: BalanceConfig;
  private readonly filters: FilterConfig;

  constructor(balance: BalanceConfig, filters: FilterConfig) {
    this.balance = balance;
    this.filters = filters;
  }

  private categorize(sample: TrainingSample): 0 | 1 | 2 {
    const val = Math.sign(sample.final_value ?? 0);
    if (val > 0) return 2; // win
    if (val < 0) return 0; // loss
    return 1; // draw
  }

  private passesFilters(sample: TrainingSample): boolean {
    const meta = sample.meta;
    const source = meta?.source?.toLowerCase();
    if (this.filters.includeSources && this.filters.includeSources.size > 0) {
      if (!source || !this.filters.includeSources.has(source)) {
        this.rejects.inc("source");
        return false;
      }
    }
    if (this.filters.excludeSources && this.filters.excludeSources.size > 0) {
      if (source && this.filters.excludeSources.has(source)) {
        this.rejects.inc("source");
        return false;
      }
    }

    if (typeof this.filters.minMoveIndex === "number" && typeof meta?.moveIndex === "number") {
      if (meta.moveIndex < this.filters.minMoveIndex) {
        this.rejects.inc("move");
        return false;
      }
    }
    if (typeof this.filters.maxMoveIndex === "number" && typeof meta?.moveIndex === "number") {
      if (meta.moveIndex > this.filters.maxMoveIndex) {
        this.rejects.inc("move");
        return false;
      }
    }
    if (typeof this.filters.minTotalMoves === "number" && typeof meta?.totalMoves === "number") {
      if (meta.totalMoves < this.filters.minTotalMoves) {
        this.rejects.inc("move");
        return false;
      }
    }
    if (typeof this.filters.maxTotalMoves === "number" && typeof meta?.totalMoves === "number") {
      if (meta.totalMoves > this.filters.maxTotalMoves) {
        this.rejects.inc("move");
        return false;
      }
    }

    if (typeof this.filters.minPolicyEntropy === "number" || typeof this.filters.maxPolicyEntropy === "number") {
      const policy = Array.isArray(sample.teacher_policy) && sample.teacher_policy.length > 0
        ? sample.teacher_policy
        : sample.mcts_policy;
      if (policy && policy.length > 0) {
        const sum = policy.reduce((acc, v) => (Number.isFinite(v) && v > 0 ? acc + v : acc), 0);
        const normalized = sum > 0 ? policy.map((v) => (Number.isFinite(v) && v > 0 ? v / sum : 0)) : undefined;
        const entropy = normalized ? shannonEntropy(normalized.filter((v) => v > 0)) : 0;
        if (normalized && typeof this.filters.minPolicyEntropy === "number" && entropy < this.filters.minPolicyEntropy) {
          this.rejects.inc("policy");
          return false;
        }
        if (normalized && typeof this.filters.maxPolicyEntropy === "number" && entropy > this.filters.maxPolicyEntropy) {
          this.rejects.inc("policy");
          return false;
        }
      }
    }

    return true;
  }

  private passesBalance(category: 0 | 1 | 2): boolean {
    if (!this.balance.enabled) return true;
    const [lossCount, drawCount, winCount] = this.balanceCounts;
    const currentCounts: [number, number, number] = [lossCount, drawCount, winCount];
    currentCounts[category] += 1;
    const total = this.totalAccepted + 1;
    const ratios = this.balance.ratios;
    const tolerance = this.balance.tolerance;
    for (let i = 0; i < 3; i++) {
      const observed = currentCounts[i] / total;
      if (observed > ratios[i] + tolerance) {
        this.rejects.inc("balance");
        return false;
      }
    }
    this.balanceCounts[category] += 1;
    return true;
  }

  accept(sample: TrainingSample): boolean {
    this.totalSeen++;
    if (!this.passesFilters(sample)) {
      return false;
    }
    const category = this.categorize(sample);
    if (!this.passesBalance(category)) {
      return false;
    }
    this.totalAccepted++;
    return true;
  }

  stats(): ReplaySamplerStats {
    return {
      seen: this.totalSeen,
      accepted: this.totalAccepted,
      rejected: this.totalSeen - this.totalAccepted,
      filtered: this.rejects.toArray(),
      finalValueCounts: {
        "-1": this.balanceCounts[0],
        "0": this.balanceCounts[1],
        "1": this.balanceCounts[2],
      },
    };
  }
}

function parseSourceSet(value: string | undefined): Set<string> | undefined {
  if (!value) return undefined;
  const items = value.split(/[ ,]+/).map((s) => s.trim().toLowerCase()).filter(Boolean);
  if (items.length === 0) return undefined;
  return new Set(items);
}

function parseNumber(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

export function createReplaySamplerFromEnv(): ReplaySampler {
  const balanceEnabled = parseBoolean(process.env.BALANCE_FINAL_VALUE, false);
  const ratios = normalizeRatios(parseRatios(process.env.BALANCE_FINAL_VALUE_RATIOS));
  const tolerance = Number(process.env.BALANCE_TOLERANCE ?? 0.05);
  const balance: BalanceConfig = {
    enabled: balanceEnabled,
    ratios,
    tolerance: Number.isFinite(tolerance) ? Math.max(0, tolerance) : 0.05,
  };

  const filters: FilterConfig = {
    minMoveIndex: parseNumber(process.env.FILTER_MIN_MOVE_INDEX),
    maxMoveIndex: parseNumber(process.env.FILTER_MAX_MOVE_INDEX),
    minTotalMoves: parseNumber(process.env.FILTER_MIN_TOTAL_MOVES),
    maxTotalMoves: parseNumber(process.env.FILTER_MAX_TOTAL_MOVES),
    includeSources: parseSourceSet(process.env.FILTER_SOURCES_INCLUDE),
    excludeSources: parseSourceSet(process.env.FILTER_SOURCES_EXCLUDE),
    minPolicyEntropy: parseNumber(process.env.FILTER_MIN_POLICY_ENTROPY),
    maxPolicyEntropy: parseNumber(process.env.FILTER_MAX_POLICY_ENTROPY),
  };

  return new ReplaySampler(balance, filters);
}





