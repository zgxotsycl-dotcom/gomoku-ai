// Simple socket.io public matchmaking test
// Connects two clients, authenticates, joins public queue and waits for 'game-start'

const { io } = require('socket.io-client');

const URL = process.env.SOCKET_URL || 'http://localhost:3002';
const PATH = '/socket.io/';

function makeClient(id, username) {
  const sock = io(URL, { path: PATH, transports: ['websocket'] });
  let started = false;
  sock.on('connect', () => {
    console.log(`[${id}] connected`);
    sock.emit('authenticate', id);
  });
  sock.on('user-counts-update', (c) => {
    console.log(`[${id}] counts`, c);
  });
  sock.on('game-start', (game) => {
    started = true;
    console.log(`[${id}] game-start`, JSON.stringify(game));
  });
  sock.on('room-created', (roomId) => {
    console.log(`[${id}] room-created`, roomId);
  });
  sock.on('disconnect', () => {
    console.log(`[${id}] disconnected`);
  });
  return {
    sock,
    joinQueue(profileId) {
      const profile = { id: profileId, username };
      sock.emit('join-public-queue', profile);
    },
    hasStarted: () => started,
    close: () => sock.close()
  };
}

async function main() {
  const c1 = makeClient('u1', 'Tester1');
  const c2 = makeClient('u2', 'Tester2');

  // Give time to connect
  await new Promise(r => setTimeout(r, 1000));
  c1.joinQueue('u1');
  c2.joinQueue('u2');

  const start = Date.now();
  while (Date.now() - start < 15000) {
    if (c1.hasStarted() || c2.hasStarted()) break;
    await new Promise(r => setTimeout(r, 250));
  }

  const ok = c1.hasStarted() || c2.hasStarted();
  console.log(`RESULT: ${ok ? 'OK' : 'FAIL'}`);
  c1.close();
  c2.close();
  process.exit(ok ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });

