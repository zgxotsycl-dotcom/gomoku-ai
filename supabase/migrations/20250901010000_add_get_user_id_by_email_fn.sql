create or replace function public.get_user_id_by_email(user_email text)
returns uuid as $$
declare
  user_id uuid;
begin
  select id into user_id from auth.users where email = user_email;
  return user_id;
end;
$$ language plpgsql security definer;
