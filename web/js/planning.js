/** District planning: the telemedicine capacity simulation. */
import { $, api, busy, errBox, num, skeleton } from './api.js';

const WINDOW_DAYS = { emergency: 7, urgent: 28, soon: 180, routine: 365 };

export function init() {
  $('#btn-sim').onclick = run;
}

function run() {
  return busy($('#btn-sim'), 'Simulating', async () => {
    const qs = new URLSearchParams({
      phcs: $('#s-phcs').value,
      ophthalmologists: $('#s-ophth').value,
      days: $('#s-days').value,
      screenings_per_phc_per_day: $('#s-rate').value,
      ai_sensitivity_referable: $('#s-sens').value,
      ai_specificity_referable: $('#s-spec').value,
    });
    $('#simout').innerHTML =
      `<div class="panel"><div class="panel-b">${skeleton('70px')}</div></div>`;
    try {
      render(await api(`/api/programme/simulate?${qs}`));
    } catch (e) {
      $('#simout').innerHTML = errBox(e.message);
    }
  });
}

function metric(value, label) {
  return `<div class="metric"><div class="v">${value}</div><div class="l">${label}</div></div>`;
}

function bar(v, colour) {
  if (v == null) return '<div class="bar"><div class="n">—</div></div>';
  return `<div class="bar"><div class="t"><i style="width:${v * 100}%;background:${colour}"></i></div>
    <div class="n">${(v * 100).toFixed(0)}%</div></div>`;
}

function render(d) {
  const s = d.summary;

  let html = `<div class="metrics rise">
    ${metric(num(s.screened), 'patients screened')}
    ${metric(`${(s.read_workload_reduction * 100).toFixed(0)}%`, 'specialist reading avoided')}
    ${metric(num(s.missed_referable_ai), 'referable cases missed')}
    ${metric(num(s.false_referrals_ai), 'false referrals generated')}</div>`;

  const rows = ['emergency', 'urgent', 'soon', 'routine'].map((k) => {
    const ai = d.ai.within_window[k];
    const manual = d.manual.within_window[k];
    if (ai == null && manual == null) return '';
    return `<div class="cmp"><div class="lbl">${k}<s>within ${WINDOW_DAYS[k]} d</s></div>
      ${bar(ai, 'var(--ok)')}${bar(manual, 'var(--bad)')}</div>`;
  }).join('');

  html += `<div class="panel rise" style="--i:1;margin-top:18px">
    <div class="panel-h"><h3>Reaching a specialist inside the clinical window</h3></div>
    <div class="panel-b">
      <div class="cmp" style="padding-bottom:6px"><div></div>
        <div class="head">AI triage</div><div class="head">Manual reading</div></div>
      ${rows}
      <p class="hint">Patients still queued when the run ended count as breaches
        rather than being excluded — otherwise an arm that never reaches them
        scores a perfect 100%.</p></div></div>`;

  const kv = (k, v) => `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  html += `<div class="panel rise" style="--i:2">
    <div class="panel-h"><h3>Specialist workload &amp; backlog</h3></div>
    <div class="panel-b"><div class="rows">
      ${kv('Image reads required — AI', num(s.specialist_reads_ai))}
      ${kv('Image reads required — manual', num(s.specialist_reads_manual))}
      ${kv('Mean wait, urgent + emergency — AI', `${s.urgent_mean_wait_days_ai ?? '—'} d`)}
      ${kv('Mean wait, urgent + emergency — manual', `${s.urgent_mean_wait_days_manual ?? '—'} d`)}
      ${kv('Backlog at close — AI', num(s.final_backlog_ai))}
      ${kv('Backlog at close — manual', num(s.final_backlog_manual))}</div>
    <p class="hint">The AI arm still carries a backlog. The difference is what is
      in it: routine patients who can safely wait, rather than the emergency
      cases stuck behind them in the manual arm.</p></div></div>`;

  $('#simout').innerHTML = html;
}
