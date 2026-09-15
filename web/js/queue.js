/** Ophthalmologist review queue and sign-off. */
import {
  $, api, busy, errBox, esc, GRADE_COLOURS as GC, panel, skeleton, state,
} from './api.js';

let items = [];

export function init() {
  $('#btn-refresh-queue').onclick = load;
}

export async function load() {
  const body = $('#queue-body');
  body.innerHTML = skeleton('60px');
  try {
    items = await api('/api/review/queue');
    if (!items.length) {
      body.innerHTML = `<div class="empty"><h4>Queue is clear</h4>
        <p>No cases are waiting for a clinician right now.</p></div>`;
      return;
    }
    body.innerHTML = items.map((it) => `
      <div class="qitem" data-id="${it.screening.id}" style="cursor:pointer">
        <div><span class="pill ${it.screening.urgency}">${esc(it.screening.urgency)}</span></div>
        <div>
          <div class="who2">${esc(it.patient_name)}</div>
          <div class="sub">${esc(it.patient_mrn)} · ${it.screening.eye} ·
            grade ${it.screening.grade ?? '—'}</div>
        </div>
        <div style="text-align:right">
          <div class="sub ${it.breaching ? 'breach' : ''}">${it.waiting_days} d waiting</div>
          ${it.breaching ? '<div class="sub breach">past window</div>' : ''}
        </div>
      </div>`).join('');
    body.querySelectorAll('.qitem').forEach((el) => {
      el.onclick = () => open(items.find((i) => i.screening.id === el.dataset.id));
    });
  } catch (e) {
    body.innerHTML = errBox(e.message);
  }
}

function open(item) {
  const sc = item.screening;
  // Sign-off is ophthalmologist-only. The server enforces this regardless;
  // hiding the control just avoids offering an action that will be refused.
  const canSign = state.user?.role === 'ophthalmologist';

  $('#queue-detail').innerHTML = panel('', `${esc(item.patient_name)} · ${sc.eye}`, `
    <div class="readout">
      <div class="score" style="color:${GC[sc.grade]}">${sc.grade ?? '—'}</div>
      <div style="padding-top:2px">
        <span class="pill ${sc.urgency}">${esc(sc.urgency)}</span>
        <div class="hint" style="margin-top:5px">${esc(sc.model_version)} ·
          ${((sc.confidence ?? 0) * 100).toFixed(0)}% confidence</div>
      </div>
    </div>
    <ul class="narr" style="margin-top:14px">${(sc.explanation || []).map((x) =>
      `<li>${esc(x)}</li>`).join('')}</ul>
    <div class="rule"></div>
    ${canSign ? `
      <div class="field"><label class="f" for="corrected">Your grade</label>
        <select id="corrected">${[0, 1, 2, 3, 4].map((g) =>
          `<option value="${g}" ${g === sc.grade ? 'selected' : ''}>Grade ${g}</option>`).join('')}</select></div>
      <div class="field"><label class="f" for="notes">Notes</label>
        <input type="text" id="notes" placeholder="Optional"></div>
      <button class="btn" id="btn-sign" style="margin-top:14px">Sign off</button>
      <div id="sign-state"></div>`
    : `<p class="hint">This session's role can read the queue but not sign off.
         Clinical sign-off is restricted to ophthalmologists.</p>`}`);

  const signBtn = $('#btn-sign');
  if (signBtn) {
    signBtn.onclick = () => busy(signBtn, 'Recording', async () => {
      try {
        await api(`/api/review/${sc.id}`, {
          method: 'POST',
          body: {
            corrected_grade: parseInt($('#corrected').value, 10),
            notes: $('#notes').value || null,
          },
        });
        $('#sign-state').innerHTML = errBox('Decision recorded.', true);
        load();
      } catch (e) {
        $('#sign-state').innerHTML = errBox(e.message);
      }
    });
  }
}
