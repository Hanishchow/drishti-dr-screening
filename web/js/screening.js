/** Screening view: patient registration, capture, result panels, history. */
import {
  $, api, busy, errBox, esc, GRADE_COLOURS as GC, panel, skeleton, state,
} from './api.js';

const LESION_LEGEND = [
  ['MA', 'Microaneurysm', '#f0503c'],
  ['HEM', 'Haemorrhage', '#a01e28'],
  ['EX', 'Hard exudate', '#f0d228'],
  ['CWS', 'Cotton-wool spot', '#5ac8c8'],
];

export function init() {
  $('#btn-patient').onclick = onFindPatient;
  $('#btn-screen').onclick = onScreen;
}

/* ------------------------------------------------------------- patient */
function onFindPatient() {
  return busy($('#btn-patient'), 'Looking up', async () => {
    const mrn = $('#mrn').value.trim();
    const name = $('#pname').value.trim();
    if (!mrn || !name) {
      $('#patient-state').innerHTML = errBox('Programme ID and name are both required.');
      return;
    }
    try {
      const yob = parseInt($('#yob').value, 10);
      state.patient = await api('/api/patients', {
        method: 'POST',
        body: { mrn, name, year_of_birth: Number.isFinite(yob) ? yob : null },
      });
      $('#patient-state').innerHTML = errBox(
        `Ready: ${state.patient.name} (${state.patient.mrn})`, true);
      $('#btn-screen').disabled = false;
      loadHistory();
    } catch (e) {
      $('#patient-state').innerHTML = errBox(e.message);
    }
  });
}

export async function loadHistory() {
  if (!state.patient) return;
  try {
    const h = await api(`/api/patients/${state.patient.id}/history`);
    if (!h.timeline.length) { $('#card-history').hidden = true; return; }
    $('#card-history').hidden = false;
    $('#history').innerHTML =
      `<thead><tr><th>Date</th><th>Eye</th><th class="n">Machine</th>
        <th class="n">Final</th><th>Urgency</th></tr></thead><tbody>` +
      h.timeline.map((t) => `<tr>
        <td class="mono">${new Date(t.captured_at).toLocaleDateString('en-IN')}</td>
        <td>${t.eye}</td>
        <td class="n" style="color:${t.grade != null ? GC[t.grade] : 'inherit'}">${t.grade ?? '—'}</td>
        <td class="n" style="color:${t.final_grade != null ? GC[t.final_grade] : 'inherit'}">${t.final_grade ?? '—'}</td>
        <td>${t.urgency ? `<span class="pill ${t.urgency}">${t.urgency}</span>` : '—'}</td>
      </tr>`).join('') + '</tbody>';
    $('#progression').innerHTML = h.progression_note
      ? `<p class="hint" style="${h.progressed ? 'color:var(--bad);font-weight:500' : ''}">
           ${esc(h.progression_note)}</p>` : '';
  } catch {
    $('#card-history').hidden = true;
  }
}

/* ----------------------------------------------------------- screening */
function onScreen() {
  return busy($('#btn-screen'), 'Screening', async () => {
    const file = $('#file').files[0];
    if (!file) {
      $('#results').innerHTML = errBox('Choose a fundus image first.');
      return;
    }
    showPending();
    const fd = new FormData();
    fd.append('patient_id', state.patient.id);
    fd.append('eye', $('#eye').value);
    // Minted client-side so a retry over a bad link is deduplicated upstream
    // rather than creating a second clinical record.
    fd.append('client_uuid', crypto.randomUUID());
    fd.append('file', file);
    try {
      state.screening = await api('/api/screenings', { method: 'POST', body: fd, form: true });
      render(state.screening, file);
      loadHistory();
    } catch (e) {
      $('#results').innerHTML = errBox(e.message);
      $('#canvas').innerHTML = '<div class="empty"><h4>Not analysed</h4></div>';
    }
  });
}

function showPending() {
  $('#canvas').innerHTML = '<div class="sk sk-img"></div>';
  $('#legend').innerHTML = '';
  $('#results').innerHTML = `<div class="panel"><div class="panel-b">
    ${skeleton('15px', '42%')}
    ${skeleton('40px', '100%', 'margin-top:14px')}
    ${skeleton('15px', '66%', 'margin-top:18px')}
    ${skeleton('15px', '52%', 'margin-top:9px')}</div></div>`;
}

export function render(sc, file) {
  if (file) {
    $('#canvas').innerHTML = `<img src="${URL.createObjectURL(file)}" alt="Fundus capture">`;
    $('#segbar').innerHTML = '';
  }
  $('#legend').innerHTML = sc.lesion_counts
    ? LESION_LEGEND.map(([k, n, c]) =>
      `<span class="it"><i class="ring" style="border-color:${c}"></i>${n}
        <b>${sc.lesion_counts[k] ?? 0}</b></span>`).join('')
    : '';

  let html = qualityPanel(sc);

  if (sc.status === 'quality_failed') {
    html += panel('', 'Not graded', `
      <p style="font-size:13px;line-height:1.6;color:var(--ink-2)">
        The capture failed the quality gate, so no grade was produced. Retake the
        photograph now, while the patient is still present — no clinical decision
        should rest on an unreadable image.</p>`, 1);
    $('#results').innerHTML = html;
    return;
  }

  html += gradePanel(sc) + triagePanel(sc) + reasoningPanel(sc);
  if (sc.latency_ms) {
    html += panel('', 'Latency', `<div class="rows">${
      Object.entries(sc.latency_ms).map(([k, v]) =>
        `<div class="row"><span class="k">${esc(k)}</span><span class="v">${v} ms</span></div>`).join('')
    }</div>`, 4);
  }
  $('#results').innerHTML = html;
}

function qualityPanel(sc) {
  const pass = sc.quality_passed;
  return panel('01', 'Image quality gate', `
    <div class="readout">
      <div class="score" style="color:${pass ? 'var(--ok)' : 'var(--bad)'}">${sc.quality_score ?? '—'}</div>
      <div style="padding-top:2px">
        <span class="pill ${pass ? 'pass' : 'fail'}">${pass ? 'Pass' : 'Recapture'}</span>
        <div class="hint" style="margin-top:5px">Capture quality, 0–100</div>
      </div>
    </div>
    <div class="track" style="margin-top:13px">
      <i style="width:${sc.quality_score ?? 0}%;background:${pass ? 'var(--ok)' : 'var(--bad)'}"></i></div>
    <p class="hint">${esc(sc.quality_guidance)}</p>
    ${(sc.quality_failures || []).map((f) =>
      `<span class="kind" style="margin-right:5px">${esc(f)}</span>`).join('')}`, 0);
}

function gradePanel(sc) {
  return panel('02', 'Severity grade', `
    <div class="readout">
      <div class="score" style="color:${GC[sc.grade]}">${sc.grade}</div>
      <div style="padding-top:2px">
        <div style="font-weight:700;font-size:14.5px">ICDR grade ${sc.grade}</div>
        <div class="hint">${((sc.confidence ?? 0) * 100).toFixed(0)}% confidence ·
          P(referable) ${((sc.referable_probability ?? 0) * 100).toFixed(0)}%</div>
      </div>
    </div>
    <div class="dist">${(sc.grade_distribution || []).map((p, i) =>
      `<i style="width:${p * 100}%;background:${GC[i]}"></i>`).join('')}</div>
    <div class="hint" style="margin-top:0">Probability mass across grades 0–4</div>
    <div class="rows" style="margin-top:14px">
      <div class="row"><span class="k">Model version</span>
        <span class="v">${esc(sc.model_version)}</span></div>
      <div class="row"><span class="k">Pipeline</span>
        <span class="v">${esc(sc.pipeline_version)}</span></div>
    </div>`, 1);
}

function triagePanel(sc) {
  return panel('03', 'Referral triage', `
    <div style="display:flex;align-items:center;gap:11px;flex-wrap:wrap">
      <span class="pill ${sc.urgency}">${esc(sc.urgency)}</span>
      <span style="font-weight:700;font-size:13.5px">Review within
        <span class="mono">${sc.days_to_review}</span> days</span>
    </div>
    ${sc.needs_human_review ? `<div class="err" style="margin-top:12px">
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
        <path d="M12 3l9 16H3z"/><path d="M12 9.5v4M12 16.5h.01"/></svg>
      <div>Routed to the review queue for a clinician.</div></div>` : ''}
    <ul class="narr" style="margin-top:13px">${(sc.triage_reasons || []).map((x) =>
      `<li class="${/Escalated/.test(x) ? 'key' : ''}">${esc(x)}</li>`).join('')}</ul>`, 2);
}

function reasoningPanel(sc) {
  return panel('04', 'Why this result', `
    <ul class="narr">${(sc.explanation || []).map((x) => {
      const alert = /Trust check FAILED|Caution/.test(x);
      const key = /^Assessment|^Measured findings|Macular involvement/.test(x);
      return `<li class="${alert ? 'alert' : (key ? 'key' : '')}">${esc(x)}</li>`;
    }).join('')}</ul>`, 3);
}
