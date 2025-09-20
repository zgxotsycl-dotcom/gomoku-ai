module.exports = {
  apps : [
    {
      name   : "gomoku-app",
      script : "npm",
      args   : "start",
      cwd    : "./gomoku-app-v2",
    },
    {
      name   : "gomoku-socket-server",
      script : "node",
      args   : "server.js",
      cwd    : "./gomoku-app-v2",
    }
  ]
}