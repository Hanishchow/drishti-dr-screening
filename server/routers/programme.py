"""District programme planning: the telemedicine capacity simulation.

Kept behind authentication like everything else, but deliberately available to
any signed-in role -- an ASHA worker seeing why urgent cases jump the queue is
a feature, not a leak. It reads no patient data at all.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..models import User
from ..security import current_user

router = APIRouter(prefix="/api/programme", tags=["programme"])


@router.get("/simulate")
def simulate(phcs: int = Query(25, ge=1, le=500),
             ophthalmologists: int = Query(2, ge=1, le=200),
             days: int = Query(180, ge=30, le=1095),
             screenings_per_phc_per_day: int = Query(12, ge=1, le=200),
             ai_sensitivity_referable: float = Query(0.92, ge=0.5, le=1.0),
             ai_specificity_referable: float = Query(0.88, ge=0.5, le=1.0),
             _: User = Depends(current_user)):
    """Run both arms -- AI triage vs every image read by an ophthalmologist --
    on one patient stream, and return the comparison."""
    from sim.district import Config, compare
    cfg = Config(phcs=phcs, ophthalmologists=ophthalmologists, days=days,
                 screenings_per_phc_per_day=screenings_per_phc_per_day,
                 ai_sensitivity_referable=ai_sensitivity_referable,
                 ai_specificity_referable=ai_specificity_referable)
    return compare(cfg)
