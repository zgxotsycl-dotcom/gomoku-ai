
alter table profiles
add column elo_rating integer default 1200,
add column is_supporter boolean default false,
add column nickname_color text,
add column badge_color text;
