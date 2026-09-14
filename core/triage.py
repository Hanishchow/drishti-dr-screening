"""Referral triage.

Converts a severity grade plus its supporting evidence into an operational
decision: who needs to be seen, how soon, and who can be told to come back next
year. This is the stage that turns a classifier into a screening programme.

Referral windows follow the pattern used by India's National Programme for
Control of Blindness DR guidelines and the ICDR severity scale.
"""
from dataclasses import dataclass, asdict

# grade -> (urgency, days_to_review, action)
BASE_PATHWAY = {
    0: ("routine", 365, "Annual re-screening at the local PHC."),
    1: ("routine", 365, "Annual re-screening; reinforce glycaemic control."),
    2: ("soon", 180, "Ophthalmologist review within 6 months."),
    3: ("urgent", 28, "Ophthalmologist review within 4 weeks; high risk of progression."),
    4: ("emergency", 7, "Immediate referral to a vitreoretinal service within 1 week."),
}

URGENCY_RANK = {"routine": 0, "soon": 1, "urgent": 2, "emergency": 3}
RANK_URGENCY = {v: k for k, v in URGENCY_RANK.items()}


@dataclass
class TriageDecision:
    urgency: str
    days_to_review: int
    action: str
    reasons: list
    needs_human_review: bool
    escalations: list

    def to_dict(self):
        return asdict(self)


def decide(grade, values, fused, quality=None, agreement=None,
           low_confidence=0.45):
    """Base pathway from the grade, then escalate on evidence the grade alone
    does not capture."""
    urgency, days, action = BASE_PATHWAY[int(grade)]
    reasons = [f"ICDR grade {grade} maps to '{urgency}' review within {days} days."]
    escalations = []
    rank = URGENCY_RANK[urgency]

    # Macular involvement is sight-threatening at ANY severity grade: a grade-2
    # eye with exudates at the fovea will lose central vision before a grade-3
    # eye with peripheral disease.
    if int(values.get("macula_ex_count", 0)) > 0 and rank < URGENCY_RANK["urgent"]:
        rank = URGENCY_RANK["urgent"]
        days = min(days, 28)
        escalations.append("macular_exudate")
        reasons.append(
            f"Escalated: {int(values['macula_ex_count'])} exudate(s) within 1500 um "
            "of the fovea indicate possible clinically significant macular oedema, "
            "which is sight-threatening independently of the severity grade.")

    if int(values.get("quadrants_with_hem", 0)) >= 4 and rank < URGENCY_RANK["urgent"]:
        rank = URGENCY_RANK["urgent"]
        days = min(days, 28)
        escalations.append("four_quadrant_haemorrhage")
        reasons.append("Escalated: haemorrhage in all four quadrants (ICDR 4-2-1 rule).")

    urgency = RANK_URGENCY[rank]
    if escalations:
        action = BASE_PATHWAY[max(int(grade), 3)][2]

    # Human review is about whether the MACHINE should be trusted on this image,
    # which is a separate question from how sick the patient is.
    needs_human = False
    if quality is not None and not quality.passed:
        needs_human = True
        reasons.append("Flagged for human review: image failed the quality gate.")
    if fused.confidence < low_confidence:
        needs_human = True
        reasons.append(f"Flagged for human review: low model confidence "
                       f"({fused.confidence:.0%}).")
    if fused.detail.get("spread", 0) >= 2:
        needs_human = True
        reasons.append("Flagged for human review: graders disagreed by two or more "
                       "severity grades.")
    if agreement and agreement.get("verdict") == "attention_unexplained":
        needs_human = True
        reasons.append("Flagged for human review: the network's attention did not "
                       "coincide with any measured lesion.")

    return TriageDecision(urgency=urgency, days_to_review=int(days), action=action,
                          reasons=reasons, needs_human_review=needs_human,
                          escalations=escalations)
