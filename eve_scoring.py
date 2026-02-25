from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

from eve_models import Deal

@dataclass
class EVEConfig:
    weights: Dict[str, float] | None = None
    logistic_a: float = 6.0
    logistic_b: float = 0.10

    def __post_init__(self):
        if self.weights is None:
            self.weights = {"v1": 0.25, "v2": 0.20, "v3": 0.20, "v4": 0.20, "v5": 0.15}
        s = sum(self.weights.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {s}")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def logistic_score(R: float, a: float, b: float) -> float:
    return 100.0 / (1.0 + math.exp(-a * (R - b)))

def discount_factors(T: int, r: float) -> List[float]:
    return [1.0 / ((1.0 + r) ** t) for t in range(1, T + 1)]

def compute_pv_cost(deal: Deal, d: List[float]) -> float:
    pv = float(deal.investment.capex_upfront)
    for t in range(deal.meta.horizon_years):
        pv += d[t] * float(deal.investment.opex_annual[t])
    return pv

def compute_v1(deal: Deal, d: List[float]) -> float:
    v1 = deal.v1_capital_productivity
    if not v1 or v1.fcf_benefit_annual is None:
        return 0.0
    return sum(d[t] * float(v1.fcf_benefit_annual[t]) for t in range(deal.meta.horizon_years))

def compute_v2(deal: Deal, d: List[float]) -> float:
    events = deal.v2_risk_events or []
    annual_reduction = sum((e.p0 * e.L0) - (e.p1 * e.L1) for e in events)
    return sum(d[t] * annual_reduction for t in range(deal.meta.horizon_years))

def compute_v3(deal: Deal, d: List[float]) -> float:
    inits = deal.v3_initiatives or []
    return sum(d[0] * (m.prob * m.months_accel * m.monthly_profit) for m in inits)

def compute_v4(deal: Deal) -> float:
    opts = deal.v4_options or []
    raw = sum(u.prob * (u.feasibility_lift * u.npv_if_pursued + u.exercise_cost_reduction_pv) for u in opts)

    oqi = 1.0
    if deal.v4_oqi:
        A = clamp(deal.v4_oqi.flexibility, 0.0, 5.0)
        V = clamp(deal.v4_oqi.portability, 0.0, 5.0)
        D = clamp(deal.v4_oqi.data_liquidity, 0.0, 5.0)
        S = clamp(deal.v4_oqi.scalability, 0.0, 5.0)
        oqi = (A + V + D + S) / 20.0
    return oqi * raw

def compute_v5(deal: Deal, d: List[float]) -> float:
    scenarios = deal.v5_resilience or []
    annual_reduction = sum(s.p * (s.mttr0_hours - s.mttr1_hours) * s.cost_per_hour for s in scenarios)
    return sum(d[t] * annual_reduction for t in range(deal.meta.horizon_years))

def detect_double_counting(deal: Deal) -> List[str]:
    warnings = []
    v2_names = set((e.name or "").strip().lower() for e in (deal.v2_risk_events or []))
    v5_names = set((s.name or "").strip().lower() for s in (deal.v5_resilience or []))
    overlap = sorted(list(v2_names.intersection(v5_names)))
    if overlap:
        warnings.append(
            "Potential double counting for: "
            + ", ".join(overlap)
            + ". Ensure V2 captures probability/impact changes and V5 captures MTTR severity only."
        )
    return warnings

def compute_eve(deal: Deal, config: Optional[EVEConfig] = None, run_sensitivity: bool = True) -> Dict[str, Any]:
    config = config or EVEConfig()
    T = deal.meta.horizon_years
    r = deal.meta.discount_rate
    d = discount_factors(T, r)

    pv_cost = compute_pv_cost(deal, d)
    if pv_cost <= 0:
        raise ValueError("PV cost must be > 0")

    deltaV1 = compute_v1(deal, d)
    deltaV2 = compute_v2(deal, d)
    deltaV3 = compute_v3(deal, d)
    deltaV4 = compute_v4(deal)
    deltaV5 = compute_v5(deal, d)

    ratios = {k: v / pv_cost for k, v in {
        "v1": deltaV1, "v2": deltaV2, "v3": deltaV3, "v4": deltaV4, "v5": deltaV5
    }.items()}

    scores = {k: logistic_score(R, config.logistic_a, config.logistic_b) for k, R in ratios.items()}

    w = config.weights
    evi = sum(w[k] * scores[k] for k in ["v1", "v2", "v3", "v4", "v5"])

    c = {
        "v1": clamp(deal.confidence.v1, 0.0, 1.0),
        "v2": clamp(deal.confidence.v2, 0.0, 1.0),
        "v3": clamp(deal.confidence.v3, 0.0, 1.0),
        "v4": clamp(deal.confidence.v4, 0.0, 1.0),
        "v5": clamp(deal.confidence.v5, 0.0, 1.0),
    }
    evi_conf = sum(w[k] * (c[k] * scores[k]) for k in ["v1", "v2", "v3", "v4", "v5"])
    confidence_weighted = sum(w[k] * c[k] for k in ["v1", "v2", "v3", "v4", "v5"])

    result = {
        "pv_cost": pv_cost,
        "pillar_pv_benefits": {"v1": deltaV1, "v2": deltaV2, "v3": deltaV3, "v4": deltaV4, "v5": deltaV5},
        "pillar_ratios": ratios,
        "pillar_scores": scores,
        "weights": w,
        "EVI": evi,
        "EVI_conf": evi_conf,
        "confidence_weighted": confidence_weighted,
        "warnings": detect_double_counting(deal),
        "sensitivities": [],
        "config": {"logistic_a": config.logistic_a, "logistic_b": config.logistic_b},
    }

    if run_sensitivity:
        result["sensitivities"] = run_simple_sensitivity(deal, config, base_evi=evi)

    return result

def run_simple_sensitivity(deal: Deal, config: EVEConfig, base_evi: float) -> List[Dict[str, float]]:
    scenarios: List[Tuple[str, str]] = [
        ("v1_fcf_plus10", "Scale V1 annual FCF benefits +10%"),
        ("v2_p1_minus10", "Reduce V2 post probabilities p1 by 10% (relative)"),
        ("v3_profit_plus10", "Scale V3 monthly profit +10%"),
        ("v5_costhr_plus10", "Scale V5 cost per hour +10%"),
    ]

    out = []
    for key, label in scenarios:
        d2 = deepcopy(deal)

        if key == "v1_fcf_plus10" and d2.v1_capital_productivity and d2.v1_capital_productivity.fcf_benefit_annual:
            d2.v1_capital_productivity.fcf_benefit_annual = [x * 1.10 for x in d2.v1_capital_productivity.fcf_benefit_annual]

        if key == "v2_p1_minus10" and d2.v2_risk_events:
            for e in d2.v2_risk_events:
                e.p1 = clamp(e.p1 * 0.90, 0.0, 1.0)

        if key == "v3_profit_plus10" and d2.v3_initiatives:
            for m in d2.v3_initiatives:
                m.monthly_profit *= 1.10

        if key == "v5_costhr_plus10" and d2.v5_resilience:
            for s in d2.v5_resilience:
                s.cost_per_hour *= 1.10

        evi2 = compute_eve(d2, config=config, run_sensitivity=False)["EVI"]
        out.append({"factor": key, "label": label, "delta_evi": float(evi2 - base_evi)})

    out.sort(key=lambda x: abs(x["delta_evi"]), reverse=True)
    return out
