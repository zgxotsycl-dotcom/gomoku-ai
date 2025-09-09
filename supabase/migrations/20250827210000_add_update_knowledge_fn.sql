create or replace function public.update_pattern_knowledge(
    p_pattern_hash text,
    p_wins_increment int,
    p_losses_increment int
)
returns void as $$
begin
    insert into public.ai_pattern_knowledge (pattern_hash, wins, losses)
    values (p_pattern_hash, p_wins_increment, p_losses_increment)
    on conflict (pattern_hash)
    do update set
        wins = ai_pattern_knowledge.wins + p_wins_increment,
        losses = ai_pattern_knowledge.losses + p_losses_increment;
end;
$$ language plpgsql;