/**
 * Entry point: session bootstrap, navigation, model status.
 *
 * The app opens straight into the dashboard. There is no sign-in screen; a
 * session is requested from the server, which grants one only where open
 * access was deliberately enabled. If it refuses, the page says so plainly
 * rather than presenting a form that would not help.
 */
import { $, API_BASE, ensureSession, esc, state } from './api.js';
import * as planning from './planning.js';
import * as queue from './queue.js';
import * as screening from './screening.js';

const VIEWS = ['screen', 'queue', 'sim'];
// Roles that can act on the review queue. The server enforces this too; this
// only decides whether the tab is worth showing.
const QUEUE_ROLES = ['ophthalmologist', 'district_admin'];

function showViews(active) {
  document.querySelectorAll('#nav button').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.view === active)));
  VIEWS.forEach((v) => { $(`#view-${v}`).hidden = (v !== active); });
  if (active === 'queue') queue.load();
}

function initNav() {
  document.querySelectorAll('#nav button').forEach((b) => {
    b.onclick = () => showViews(b.dataset.view);
  });
}

async function loadHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (!r.ok) return;
    const h = await r.json();
    const chip = $('#chip-model');
    chip.dataset.on = h.model.available ? '1' : '0';
    chip.lastChild.textContent = h.model.available
      ? `model ${h.model.version}` : 'no model loaded';
  } catch { /* header detail only; not worth surfacing */ }
}

function blocked() {
  $('#blocked-body').innerHTML = `
    <p style="font-size:13px;line-height:1.6;color:var(--ink-2)">
      The server did not grant a session. Authorisation is still enforced on
      every endpoint, so the dashboard cannot load without one.</p>
    <p class="hint">Start the server with <code>DRISHTI_OPEN_ACCESS=1</code> and
      <code>DRISHTI_NODE_ROLE=edge</code> for a demo or a PHC node. A district
      node refuses open access by design — obtain a token from
      <code>/api/auth/token</code> with credentials instead.</p>`;
  $('#view-blocked').hidden = false;
  $('#app').hidden = true;
  $('#nav').hidden = true;
}

function ready() {
  $('#view-blocked').hidden = true;
  $('#app').hidden = false;
  $('#nav').hidden = false;
  $('#who').hidden = false;
  $('#who').innerHTML =
    `<b>${esc(state.user.full_name)}</b> · ${esc(state.user.role.replace(/_/g, ' '))}`;
  document.querySelector('[data-view="queue"]').hidden =
    !QUEUE_ROLES.includes(state.user.role);
  showViews('screen');
}

async function boot() {
  initNav();
  screening.init();
  queue.init();
  planning.init();
  loadHealth();

  const session = await ensureSession();
  if (session) ready(); else blocked();
}

boot();
