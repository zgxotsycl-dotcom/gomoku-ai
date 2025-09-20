-- Opening book table storing canonical board hashes and best moves
create table if not exists public.ai_opening_book (
  board_hash text primary key,
  best_move smallint[] not null,
  move_count integer
);

comment on table public.ai_opening_book is 'Canonicalized opening book entries for NN-guided engine';
comment on column public.ai_opening_book.board_hash is 'Canonical board hash (symmetry-reduced)';
comment on column public.ai_opening_book.best_move is 'Best move as [row, col]';
comment on column public.ai_opening_book.move_count is 'Optional move count at which the entry applies';

