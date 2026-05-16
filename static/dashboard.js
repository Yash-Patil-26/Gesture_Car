// static/dashboard.js
// ─────────────────────────────────────────────────────────────
// WebSocket client + DOM updates for the gesture car dashboard.
// Connects to Flask-SocketIO server on same origin.
// ─────────────────────────────────────────────────────────────

const socket = io();

// ── DOM refs ───────────────────────────────────────────────────
const cmdDisplay  = document.getElementById('command-display');
const gestureLabel= document.getElementById('gesture-label');
const confBar     = document.getElementById('conf-bar');
const confValue   = document.getElementById('conf-value');
const fpsBadge    = document.getElementById('fps-badge');
const logBody     = document.getElementById('log-body');

const pillCam     = document.getElementById('pill-cam');
const pillEsp     = document.getElementById('pill-esp');
const pillWs      = document.getElementById('pill-ws');

const dpadBtns = {
  FORWARD: document.getElementById('btn-forward'),
  REVERSE: document.getElementById('btn-reverse'),
  LEFT:    document.getElementById('btn-left'),
  RIGHT:   document.getElementById('btn-right'),
  STOP:    document.getElementById('btn-stop'),
};

const statEls = {
  forward: document.getElementById('stat-forward'),
  reverse: document.getElementById('stat-reverse'),
  left:    document.getElementById('stat-left'),
  right:   document.getElementById('stat-right'),
  stop:    document.getElementById('stat-stop'),
};

// ── WebSocket connection status ────────────────────────────────
socket.on('connect', () => {
  pillWs.classList.add('online');
  pillWs.classList.remove('offline');
  pillWs.textContent = '● WebSocket';
});

socket.on('disconnect', () => {
  pillWs.classList.remove('online');
  pillWs.classList.add('offline');
  pillWs.textContent = '● WebSocket';
});

// ── State update handler ───────────────────────────────────────
let lastCmd = null;

socket.on('state_update', (data) => {
  // ── Gesture label ──────────────────────────────────────────
  gestureLabel.textContent = data.gesture || '—';

  // ── Confidence bar ─────────────────────────────────────────
  const pct = Math.round((data.confidence || 0) * 100);
  confBar.style.width    = pct + '%';
  confValue.textContent  = pct + '%';
  // Color: green if above threshold, blue if below
  confBar.style.background = pct >= 85 ? '#00e676' : '#4dabf7';

  // ── Command display ────────────────────────────────────────
  const cmd = data.command || 'STOP';
  cmdDisplay.textContent       = cmd;
  cmdDisplay.setAttribute('data-cmd', cmd);

  // ── D-pad highlight ────────────────────────────────────────
  Object.entries(dpadBtns).forEach(([key, el]) => {
    el.classList.toggle('active', key === cmd);
  });

  // ── FPS badge ──────────────────────────────────────────────
  fpsBadge.textContent = (data.fps || 0) + ' fps';

  // ── Connection pills ───────────────────────────────────────
  if (data.esp) {
    pillEsp.classList.add('online');
    pillEsp.classList.remove('offline');
  } else {
    pillEsp.classList.remove('online');
    pillEsp.classList.add('offline');
  }

  // Camera pill — always green if we're receiving data
  pillCam.classList.add('online');
  pillCam.classList.remove('offline');

  // ── Session stats ──────────────────────────────────────────
  if (data.stats) {
    Object.entries(data.stats).forEach(([gesture, count]) => {
      const el = statEls[gesture];
      if (el) el.textContent = count;
    });
  }

  // ── Command log (last 5) ───────────────────────────────────
  if (data.log && data.log.length > 0) {
    // Only re-render if command changed
    if (cmd !== lastCmd) {
      lastCmd = cmd;
      const rows = data.log.slice().reverse().map(entry => {
        const cmdColor = {
          FORWARD: '#00e676', REVERSE: '#4dabf7',
          LEFT: '#b39ddb',   RIGHT: '#ffb300', STOP: '#ff5252'
        }[entry.cmd] || '#e8eaf0';

        return `
          <tr>
            <td>${entry.time}</td>
            <td style="color:${cmdColor};font-weight:700">${entry.cmd}</td>
            <td>${Math.round(entry.conf * 100)}%</td>
          </tr>`;
      }).join('');

      logBody.innerHTML = rows || '<tr><td colspan="3" class="empty-log">No commands yet</td></tr>';
    }
  }
});

// ── Periodic status poll (fallback if WebSocket drops) ────────
setInterval(() => {
  fetch('/status')
    .then(r => r.json())
    .then(data => {
      // Only use as fallback — WebSocket is primary
      if (!socket.connected) {
        gestureLabel.textContent = data.gesture || '—';
        cmdDisplay.textContent   = data.command || 'STOP';
        fpsBadge.textContent     = (data.fps || 0) + ' fps';
      }
    })
    .catch(() => {});
}, 2000);