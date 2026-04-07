/* ── Meeting RL Agent – Frontend ──────────────────────── */

let ws = null;
let selectedTask = 1;
let rewardH = [], labels = [];
let trained = false;

/* ── Task Selection ─────────────────────────────────────── */
function selectTask(n) {
  selectedTask = n;
  document.querySelectorAll('.task-card').forEach(c => {
    c.classList.toggle('selected', parseInt(c.dataset.task) === n);
  });
}

/* ── WebSocket ──────────────────────────────────────────── */
function connect() {
  const p = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${p}://${location.host}/ws`);
  ws.onopen = () => setStatus('online', 'Connected');
  ws.onclose = () => { setStatus('offline', 'Disconnected'); setTimeout(connect, 2000); };
  ws.onerror = () => setStatus('offline', 'Error');
  ws.onmessage = e => handle(JSON.parse(e.data));
}

function handle(msg) {
  switch (msg.type) {
    case 'episode': onEpisode(msg.data); break;
    case 'log': addLogLine(msg.line, msg.cls); break;
    case 'training_complete': onDone(); break;
    case 'training_stopped': onDone(); break;
    case 'reset_done': onResetDone(); break;
    case 'state_result': onStateResult(msg); break;
    case 'error': alert(msg.message); break;
  }
}

function addLogLine(text, cls = '') {
  const log = document.getElementById('live-log');
  if (!log) return;
  const line = document.createElement('span');
  line.className = `log-line ${cls}`;
  line.textContent = text;
  log.appendChild(line);
  // Auto scroll to bottom
  log.scrollTop = log.scrollHeight;
  // Keep only last 100 lines for performance
  if (log.childNodes.length > 100) log.removeChild(log.firstChild);
}

/* ── Episode Handler (live training) ────────────────────── */
function onEpisode(d) {
  rewardH.push(d.total_reward);
  labels.push(d.total_steps);
  const pb = document.getElementById("progress-bar");
  const pl = document.getElementById("progress-label");

  const setT = (id, txt) => {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  };

  const w = Math.min(20, rewardH.length);
  const avg = arr => arr.slice(-w).reduce((a, b) => a + b, 0) / w;

  setT('v-steps', d.total_steps);
  setT('v-steps-live', d.total_steps); // secondary ID just in case
  setT('v-rw', avg(rewardH).toFixed(2));
  setT('v-sc', Math.round(d.scheduled / d.total_meetings * 100) + '%');
  setT('v-cf', d.conflicts);
  setT('v-eps', d.epsilon.toFixed(4));

  const pct = (d.total_steps / 20 * 100);
  if (pb) pb.style.width = pct + '%';
  if (pl) pl.textContent = `Step ${d.total_steps} / 20`;
}

function onDone() {
  trained = true;
  setStatus('online', 'Done (Training Complete)');
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-state').disabled = false;
}

/* ── Reset ──────────────────────────────────────────────── */
function doReset() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'reset' }));
}

function onResetDone() {
  trained = false;
  rewardH = [];
  labels = [];

  // Hide panels
  document.getElementById('live-panel').classList.add('hidden');
  document.getElementById('state-panel').classList.add('hidden');
  document.getElementById('prog-wrap').classList.add('hidden');

  // Reset metrics
  document.getElementById('v-steps').textContent = '0';
  document.getElementById('v-rw').textContent = '0.00';
  document.getElementById('v-sc').textContent = '0%';
  document.getElementById('v-cf').textContent = '0';
  document.getElementById('v-eps').textContent = '1.00';

  // Reset buttons
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-state').disabled = true;

  // Reset task selection to 1
  selectTask(1);

  setStatus('online', 'Connected');
}

/* ── Start Training ─ ────────────────────────────────────── */
function doStart() {
  if (!ws || ws.readyState !== 1) return;

  // Reset live data
  rewardH = [];
  labels = [];

  // Show live panel, hide state panel
  document.getElementById('live-log').innerHTML = ''; // Clear logs
  document.getElementById('live-panel').classList.remove('hidden');
  document.getElementById('state-panel').classList.add('hidden');
  document.getElementById('prog-wrap').classList.remove('hidden');
  document.getElementById('prog-bar').style.width = '0%';
  document.getElementById('prog-label').textContent = '';

  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-state').disabled = true;

  setStatus('busy', 'Training…');

  ws.send(JSON.stringify({
    type: 'train',
    task: selectedTask,
    episodes: 1,
    num_meetings: +document.getElementById('in-meet').value || 5,
    lr: 0.15,
    gamma: 0.95,
    epsilon: 1.0,
    epsilon_decay: 0.995,
  }));
}

/* ── State (show full results) ──────────────────────────── */
function doState() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'state' }));
}

function onStateResult(msg) {
  if (!msg.data) {
    alert(msg.message || 'No data available');
    return;
  }

  const d = msg.data;

  // Hide live panel, show state panel
  document.getElementById('live-panel').classList.add('hidden');
  document.getElementById('prog-wrap').classList.add('hidden');
  document.getElementById('state-panel').classList.remove('hidden');

  // Header
  document.getElementById('state-task-badge').textContent = `Task ${d.task}`;
  document.getElementById('state-task-name').textContent = d.task_name;
  document.getElementById('state-summary').textContent =
    `${d.total_episodes} trained runs · ${d.total_meetings_per_ep} meetings/run · ${d.q_table_size} Q-states`;

  // Performance table
  const rows = [
    ['Avg Reward (first 20 runs)', d.avg_reward_first20],
    ['Avg Reward (last 20 runs)', d.avg_reward_last20],
    ['Reward Improvement', formatDelta(d.reward_improvement)],
    ['Best Reward', d.best_reward],
    ['Worst Reward', d.worst_reward],
    ['Avg Scheduled', d.avg_scheduled_last20 + ' / ' + d.total_meetings_per_ep],
    ['Avg Conflicts', d.avg_conflicts_last20],
    ['Avg Pref Matches', d.avg_pref_matches_last20],
    ['Q-Table States', d.q_table_size],
  ];
  const tbody = document.querySelector('#perf-table tbody');
  tbody.innerHTML = '';
  rows.forEach(([label, val]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${label}</td><td>${val}</td>`;
    tbody.appendChild(tr);
  });

  // Task score
  document.getElementById('big-score').textContent = d.task_score_last20.toFixed(4);
  document.getElementById('score-first').textContent = d.task_score_first20.toFixed(4);
  document.getElementById('score-last').textContent = d.task_score_last20.toFixed(4);
  document.getElementById('score-best').textContent = d.task_score_best.toFixed(4);

  // Color the big score
  const bs = document.getElementById('big-score');
  bs.style.color = d.task_score_last20 >= 0.8 ? 'var(--green)' : d.task_score_last20 >= 0.5 ? 'var(--yellow)' : 'var(--red)';

  // Schedule
  renderSched(d.best_episode);

  // Log
  renderLog(d.best_episode);
}

/* ── Render Schedule Grid ───────────────────────────────── */
const SLOTS = ['9AM', '10AM', '11AM', '12PM', '1PM', '2PM', '3PM', '4PM'];
const SLOT_L = { '9AM': '9 AM', '10AM': '10 AM', '11AM': '11 AM', '12PM': '12 PM', '1PM': '1 PM', '2PM': '2 PM', '3PM': '3 PM', '4PM': '4 PM' };

function renderSched(d) {
  const g = document.getElementById('sched');
  const sm = d.schedule_map || {};
  const prefs = {};
  (d.actions || []).forEach(a => { if (a.info && a.info.preferred_match) prefs[a.time_slot] = true; });
  const prio = {};
  (d.actions || []).forEach(a => {
    const t = a.meeting_title || '';
    prio[a.meeting_id] = /Sprint|Client|Product|Architecture/i.test(t) ? 3 : /Design|Code|Retro|Budget/i.test(t) ? 2 : 1;
  });

  g.innerHTML = '';
  SLOTS.forEach(s => {
    const info = sm[s];
    if (info) {
      const p = prio[info.meeting_id] || 2;
      const pf = prefs[s] ? '<span class="pref">⭐</span>' : '';
      g.innerHTML += `<div class="slot filled p${p}"><span class="st">${SLOT_L[s]}</span><span class="sm">${info.title}${pf}</span></div>`;
    } else {
      g.innerHTML += `<div class="slot empty"><span class="st">${SLOT_L[s]}</span></div>`;
    }
  });
}

function renderLog(d) {
  const log = document.getElementById('log');
  if (!d.actions || !d.actions.length) { log.innerHTML = '<p class="dim">No actions</p>'; return; }
  log.innerHTML = '';
  d.actions.forEach(a => {
    const conflict = a.info && (a.info.conflict || a.info.slot_unavailable);
    const cls = a.action_type === 'reject' ? 'bad' : conflict ? 'warn' : 'ok';
    const txt = a.action_type === 'schedule'
      ? (conflict ? `⚡ ${a.meeting_title} → ${a.time_slot}` : `✓ ${a.meeting_title} → ${a.time_slot}`)
      : `✕ Reject ${a.meeting_title}`;
    const rc = a.reward >= 0 ? 'ok' : 'bad';
    log.innerHTML += `<div class="le"><span class="ls">${a.step}</span><span class="la ${cls}">${txt}</span><span class="lr ${rc}">${a.reward > 0 ? '+' : ''}${a.reward.toFixed(2)}</span></div>`;
  });
}

function formatDelta(v) {
  const sign = v >= 0 ? '+' : '';
  const color = v >= 0 ? 'var(--green)' : 'var(--red)';
  return `<span style="color:${color}">${sign}${v}</span>`;
}

function setStatus(cls, txt) {
  const el = document.getElementById('status');
  if (el) {
    el.className = 'status ' + cls;
    el.textContent = txt;
  }
}

/* ── HTTP API Tester (REST) ─────────────────────────────── */
async function drawApiOut(res) {
  const out = document.getElementById('api-out');
  try {
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
    out.style.color = res.ok ? 'black' : 'var(--red)';
  } catch (e) {
    out.textContent = `Error: HTTP ${res.status}`;
    out.style.color = 'var(--red)';
  }
}

async function apiReset() {
  document.getElementById('api-out').textContent = "Fetching GET /reset...";
  const res = await fetch('/reset', { method: 'GET' });
  drawApiOut(res);
}

async function apiState() {
  document.getElementById('api-out').textContent = "Fetching GET /state...";
  const res = await fetch('/state', { method: 'GET' });
  drawApiOut(res);
}

async function apiStep() {
  document.getElementById('api-out').textContent = "Sending POST /step...";
  const action_type = document.getElementById('api-action-type').value;
  const meeting_id = parseInt(document.getElementById('api-meeting-id').value);
  const time_slot = document.getElementById('api-time-slot').value;
  
  const payload = { action_type, meeting_id, time_slot };
  
  const res = await fetch('/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  drawApiOut(res);
}

document.addEventListener('DOMContentLoaded', connect);

/* ── Tab Slider View Switcher ─────────────────────────────── */
function switchView(view) {
  // Update Buttons
  document.getElementById('tab-task').classList.toggle('active', view === 'task');
  document.getElementById('tab-api').classList.toggle('active', view === 'api');

  // Update Panels
  if (view === 'task') {
    document.getElementById('view-task').classList.remove('hidden');
    document.getElementById('view-api').classList.add('hidden');
  } else if (view === 'api') {
    document.getElementById('view-task').classList.add('hidden');
    document.getElementById('view-api').classList.remove('hidden');
  }
}
