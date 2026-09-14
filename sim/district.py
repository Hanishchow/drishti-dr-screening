"""District-scale telemedicine simulation.

A grading model is not a screening programme. The question a district health
officer actually asks is: given N primary health centres, one ASHA worker each,
and only M ophthalmologists in the district, how many people get screened, and
do the sight-threatening cases reach a specialist in time?

This is a discrete-event simulation over working days. It runs two arms on the
same synthetic patient stream:

  manual   every captured image is queued for ophthalmologist reading
  ai       the pipeline grades and triages; only referrals and low-confidence
           or quality-failed cases consume specialist time

The comparison is the point. The AI arm is not judged on accuracy in isolation
but on whether it gets urgent patients seen sooner with the SAME specialist
headcount -- and the simulation also counts the cost of its mistakes, since a
missed referable case is a patient who silently drops out of the programme.
"""
import heapq
from dataclasses import dataclass, field, asdict

import numpy as np

# Indian DR epidemiology, used to generate a realistic grade mix.
# Roughly: of diabetics screened, most have no retinopathy; referable disease
# (grade >= 2) runs around 10-12% in community screening studies.
GRADE_PREVALENCE = [0.665, 0.175, 0.095, 0.04, 0.025]

URGENCY_WINDOW_DAYS = {"routine": 365, "soon": 180, "urgent": 28, "emergency": 7}
URGENCY_PRIORITY = {"emergency": 0, "urgent": 1, "soon": 2, "routine": 3}


@dataclass
class Config:
    phcs: int = 25                       # primary health centres in the district
    ophthalmologists: int = 2            # district specialist headcount
    days: int = 180
    working_days_per_week: int = 6
    screenings_per_phc_per_day: int = 12
    # A specialist reading images remotely is far faster than an in-person exam.
    reads_per_ophthalmologist_per_day: int = 90
    # But a referred patient still needs a real clinic slot.
    clinic_slots_per_ophthalmologist_per_day: int = 18
    recapture_rate: float = 0.18         # fraction of captures failing the gate
    max_recapture_attempts: int = 2
    ai_sensitivity_referable: float = 0.92
    ai_specificity_referable: float = 0.88
    ai_flag_for_review_rate: float = 0.10   # low confidence / quality / disagreement
    seed: int = 0


@dataclass
class Metrics:
    screened: int = 0
    ungradable: int = 0
    referred: int = 0
    seen: int = 0
    unseen: int = 0          # still queued when the run ended
    missed_referable: int = 0
    false_referrals: int = 0
    specialist_reads: int = 0
    wait_days: dict = field(default_factory=dict)
    within_window: dict = field(default_factory=dict)
    backlog_curve: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["wait_days"] = {k: (round(float(np.mean(v)), 1) if v else None)
                          for k, v in self.wait_days.items()}
        d["within_window"] = {k: (round(float(np.mean(v)), 3) if v else None)
                              for k, v in self.within_window.items()}
        return d


def _is_working_day(day, cfg):
    return (day % 7) < cfg.working_days_per_week


def _urgency_for_grade(grade):
    return {0: "routine", 1: "routine", 2: "soon", 3: "urgent", 4: "emergency"}[grade]


def simulate(cfg: Config, arm="ai"):
    """Run one arm. Returns Metrics."""
    rng = np.random.default_rng(cfg.seed)
    m = Metrics(wait_days={k: [] for k in URGENCY_WINDOW_DAYS},
                within_window={k: [] for k in URGENCY_WINDOW_DAYS})

    # Entries are (priority, day_queued, seq, patient). The seq is a strictly
    # increasing tie-breaker: without a unique third element, two entries with
    # the same priority and queue day make heapq compare the payload dicts,
    # which raises. It also makes ordering FIFO within a priority band.
    read_queue = []        # specialist remote reading
    clinic_queue = []      # in-person referral slots
    seq = 0

    def push(queue, urgency, day_queued, patient):
        nonlocal seq
        seq += 1
        heapq.heappush(queue, (URGENCY_PRIORITY[urgency], day_queued, seq, patient))

    for day in range(cfg.days):
        if not _is_working_day(day, cfg):
            m.backlog_curve.append(len(read_queue) + len(clinic_queue))
            continue

        # ---- capture stage ----
        captures = cfg.phcs * cfg.screenings_per_phc_per_day
        for _ in range(captures):
            true_grade = int(rng.choice(5, p=GRADE_PREVALENCE))
            referable = true_grade >= 2

            # Quality gate, with a bounded number of recapture attempts.
            attempts = 0
            while attempts < cfg.max_recapture_attempts and rng.random() < cfg.recapture_rate:
                attempts += 1
            if attempts >= cfg.max_recapture_attempts and rng.random() < cfg.recapture_rate:
                m.ungradable += 1
                # Ungradable images still need a human eye, in both arms.
                push(read_queue, "soon", day,
                     {"grade": true_grade, "referable": referable,
                      "urgency": "soon", "ungradable": True})
                continue

            m.screened += 1

            if arm == "manual":
                # No triage available: every image waits for a specialist read,
                # in arrival order.
                push(read_queue, "routine", day,
                     {"grade": true_grade, "referable": referable,
                      "urgency": _urgency_for_grade(true_grade),
                      "ungradable": False})
                continue

            # ---- AI arm: grade, triage, and only escalate what needs it ----
            if referable:
                detected = rng.random() < cfg.ai_sensitivity_referable
            else:
                detected = rng.random() > cfg.ai_specificity_referable
            flagged = rng.random() < cfg.ai_flag_for_review_rate

            if detected:
                pred_grade = true_grade if referable else 2
                urgency = _urgency_for_grade(pred_grade)
                m.referred += 1
                if not referable:
                    m.false_referrals += 1
                push(clinic_queue, urgency, day,
                     {"grade": true_grade, "referable": referable,
                      "urgency": urgency, "ungradable": False})
            elif flagged:
                # Not referred by the model, but the model said it was unsure --
                # this is the safety net, and it is what makes the trust checks
                # in the explainability layer operationally meaningful.
                push(read_queue, "soon", day,
                     {"grade": true_grade, "referable": referable,
                      "urgency": _urgency_for_grade(true_grade),
                      "ungradable": False})
            elif referable:
                # Missed: patient is told they are fine and leaves the programme
                # until the next annual round.
                m.missed_referable += 1

        # ---- specialist capacity ----
        reads_left = cfg.ophthalmologists * cfg.reads_per_ophthalmologist_per_day
        while read_queue and reads_left > 0:
            _, queued, _, p = heapq.heappop(read_queue)
            reads_left -= 1
            m.specialist_reads += 1
            # A remote read that finds disease converts into a clinic referral.
            if p["referable"]:
                push(clinic_queue, p["urgency"], queued, p)

        slots_left = cfg.ophthalmologists * cfg.clinic_slots_per_ophthalmologist_per_day
        while clinic_queue and slots_left > 0:
            _, queued, _, p = heapq.heappop(clinic_queue)
            slots_left -= 1
            wait = day - queued
            u = p["urgency"]
            m.seen += 1
            m.wait_days[u].append(wait)
            m.within_window[u].append(1.0 if wait <= URGENCY_WINDOW_DAYS[u] else 0.0)

        m.backlog_curve.append(len(read_queue) + len(clinic_queue))

    # End-of-run accounting. Without this, within_window is computed only over
    # patients who were actually seen, so an arm that leaves thousands of people
    # in the queue scores 100% simply because the ones it never reached are not
    # counted. Everyone still queued is charged against their window using the
    # wait they have already accrued.
    for queue in (read_queue, clinic_queue):
        for _, queued, _, p in queue:
            u = p["urgency"]
            wait = cfg.days - queued
            m.unseen += 1
            m.wait_days[u].append(wait)
            m.within_window[u].append(1.0 if wait <= URGENCY_WINDOW_DAYS[u] else 0.0)

    return m


def compare(cfg: Config):
    """Run both arms on the same configuration and summarise the difference."""
    ai = simulate(cfg, "ai")
    manual = simulate(cfg, "manual")

    def urgent_wait(m):
        vals = m.wait_days["emergency"] + m.wait_days["urgent"]
        return float(np.mean(vals)) if vals else None

    ai_w, man_w = urgent_wait(ai), urgent_wait(manual)
    return {
        "config": asdict(cfg),
        "ai": ai.to_dict(),
        "manual": manual.to_dict(),
        "summary": {
            "screened": ai.screened,
            "specialist_reads_ai": ai.specialist_reads,
            "specialist_reads_manual": manual.specialist_reads,
            "read_workload_reduction": (
                round(1 - ai.specialist_reads / max(manual.specialist_reads, 1), 3)),
            "urgent_mean_wait_days_ai": None if ai_w is None else round(ai_w, 1),
            "urgent_mean_wait_days_manual": None if man_w is None else round(man_w, 1),
            "final_backlog_ai": ai.backlog_curve[-1] if ai.backlog_curve else 0,
            "final_backlog_manual": (manual.backlog_curve[-1]
                                     if manual.backlog_curve else 0),
            "missed_referable_ai": ai.missed_referable,
            "false_referrals_ai": ai.false_referrals,
        },
    }


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    for f, v in asdict(Config()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items() if k != "json"})
    out = compare(cfg)

    if args.json:
        print(json.dumps(out, indent=2))
        return

    s = out["summary"]
    print(f"District simulation: {cfg.phcs} PHCs, {cfg.ophthalmologists} "
          f"ophthalmologists, {cfg.days} days\n")
    print(f"  patients screened            : {s['screened']:,}")
    print(f"  specialist image reads (AI)  : {s['specialist_reads_ai']:,}")
    print(f"  specialist image reads (man.): {s['specialist_reads_manual']:,}")
    print(f"  reading workload reduction   : {s['read_workload_reduction']:.1%}")
    print(f"  urgent mean wait, AI         : {s['urgent_mean_wait_days_ai']} days")
    print(f"  urgent mean wait, manual     : {s['urgent_mean_wait_days_manual']} days")
    print(f"  final backlog, AI            : {s['final_backlog_ai']:,}")
    print(f"  final backlog, manual        : {s['final_backlog_manual']:,}")
    print(f"  referable cases missed by AI : {s['missed_referable_ai']:,}")
    print(f"  false referrals by AI        : {s['false_referrals_ai']:,}")
    print("\n  % seen inside clinical window, AI arm:")
    for k in ("emergency", "urgent", "soon", "routine"):
        a = out["ai"]["within_window"].get(k)
        mn = out["manual"]["within_window"].get(k)
        if a is None and mn is None:
            continue
        fa = "-" if a is None else f"{a:.1%}"
        fm = "-" if mn is None else f"{mn:.1%}"
        print(f"    {k:10} {fa:>8} {fm:>8}")


if __name__ == "__main__":
    main()
