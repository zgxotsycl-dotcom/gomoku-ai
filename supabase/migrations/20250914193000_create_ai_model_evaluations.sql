-- Create table to log arena gating evaluations
create table if not exists public.ai_model_evaluations (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  games integer not null default 0,
  candidate_wins integer not null default 0,
  prod_wins integer not null default 0,
  draws integer not null default 0,
  winrate numeric not null default 0,
  candidate_fingerprint text,
  prod_fingerprint text,
  threshold numeric not null default 0.5
);

create index if not exists ai_model_evaluations_created_at_idx on public.ai_model_evaluations (created_at desc);

