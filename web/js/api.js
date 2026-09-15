/**
 * Transport, session and shared helpers.
 *
 * There is no sign-in form in this UI. A session is obtained from
 * /api/auth/session, which the server grants only where open access was
 * deliberately enabled. Authorisation itself is unchanged: the token carries a
 * real role, and every endpoint keeps checking capabilities, so a session
 * obtained this way still cannot sign off a grade unless its role allows it.
 */

/**
 * Where the API lives.
 *
 * Empty means same-origin, which is the case when the backend serves this
 * page. Set a `<meta name="drishti-api" content="https://...">` in index.html
 * (or `window.DRISHTI_API`) to run the frontend separately from the API --
 * which is what the `frontend` branch and any static host need.
 */
export const API_BASE = (
  document.querySelector('meta[name="drishti-api"]')?.content
  || window.DRISHTI_API
  || ''
).replace(/\/$/, '');

const url = (path) => API_BASE + path;

export const state = {
  token: null,
  user: null,
  patient: null,
  screening: null,
};

export const GRADE_COLOURS = [
  'var(--g0)', 'var(--g1)', 'var(--g2)', 'var(--g3)', 'var(--g4)',
];

export const $ = (sel) => document.querySelector(sel);

export const esc = (s) => String(s ?? '').replace(
  /[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

export const num = (n) => (n === null || n === undefined)
  ? '—' : Number(n).toLocaleString('en-IN');

/** Restore a token so a reload does not cost a round trip. */
function loadStored() {
  try {
    const raw = localStorage.getItem('drishti.session');
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function store(tok) {
  try { localStorage.setItem('drishti.session', JSON.stringify(tok)); } catch { /* private mode */ }
}

function forget() {
  try { localStorage.removeItem('drishti.session'); } catch { /* private mode */ }
}

export async function api(path, { method = 'GET', body, form, auth = true } = {}) {
  const headers = {};
  if (auth && state.token) headers.Authorization = `Bearer ${state.token}`;

  let payload = body;
  if (body && !form) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  const r = await fetch(url(path), { method, headers, body: payload });

  if (r.status === 401 && auth) {
    // The stored token refers to a user the server no longer has, or it
    // expired. Discard it and take one fresh session rather than bouncing the
    // clinician to a form that no longer exists.
    forget();
    state.token = null;
    const renewed = await openSession();
    if (renewed) return api(path, { method, body, form, auth });
    throw new Error('Session could not be renewed.');
  }

  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!r.ok) throw new Error(data?.detail || `${r.status} ${r.statusText}`);
  return data;
}

/** Ask the server for a session. Returns the token payload, or null. */
export async function openSession() {
  const r = await fetch(url('/api/auth/session'), { method: 'POST' });
  if (!r.ok) return null;
  const tok = await r.json();
  state.token = tok.access_token;
  state.user = tok;
  store(tok);
  return tok;
}

/** Reuse a stored session if present, otherwise request one. */
export async function ensureSession() {
  const saved = loadStored();
  if (saved?.access_token) {
    state.token = saved.access_token;
    state.user = saved;
    try {
      // Confirm the token still resolves to a live user before trusting it.
      await api('/api/auth/me');
      return saved;
    } catch { /* falls through to a fresh session */ }
  }
  return openSession();
}

export function can(capabilityRoles) {
  return capabilityRoles.includes(state.user?.role);
}

/* ------------------------------------------------------------- UI helpers */
export function errBox(msg, ok = false) {
  return `<div class="err ${ok ? 'ok-note' : ''}">
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
      <circle cx="12" cy="12" r="9"/><path d="M12 7.5v5M12 16h.01"/></svg>
    <div>${esc(msg)}</div></div>`;
}

export async function busy(btn, label, fn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = `<span class="spin"></span>${label}`;
  try { return await fn(); } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

export function panel(step, title, body, i = 0) {
  return `<div class="panel rise" style="--i:${i}">
    <div class="panel-h">${step ? `<span class="step">${step}</span>` : ''}<h3>${title}</h3></div>
    <div class="panel-b">${body}</div></div>`;
}

export function skeleton(height, width = '100%', extra = '') {
  return `<div class="sk" style="height:${height};width:${width};${extra}"></div>`;
}
