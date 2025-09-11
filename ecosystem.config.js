module.exports = {
  apps : [{
    name   : "gomoku-trainer",
    script : "deno",
    args   : "task train",
    cwd    : "./training_scripts",
    env    : {
      "SUPABASE_URL": "https://xkwgfidiposftwwasdqs.supabase.co",
      "SUPABASE_ANON_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE",
      "SUPABASE_SERVICE_ROLE_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTA4Mzc3MywiZXhwIjoyMDcwNjU5NzczfQ.eogwLg8qEHah1KenIWEcQMkU0ZYB3FomB4Bk0FTpcj4"
    }
  }]
}