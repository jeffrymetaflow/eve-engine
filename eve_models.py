from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, conint, confloat, model_validator

SourceType = Literal["provided", "assumed", "estimated"]

class Note(BaseModel):
    text: str
    source: SourceType = "provided"

class Company(BaseModel):
    industry: str
    revenue: Optional[float] = None
    ebitda_margin: Optional[float] = None

class Meta(BaseModel):
    company: Company
    horizon_years: conint(ge=1, le=15) = 5
    discount_rate: confloat(ge=0.0, le=0.5) = 0.10
    currency: str = "USD"

class Investment(BaseModel):
    capex_upfront: confloat(ge=0.0) = 0.0
    opex_annual: List[confloat(ge=0.0)]

class V1CapitalProductivity(BaseModel):
    fcf_benefit_annual: Optional[List[float]] = None
    notes: List[Note] = Field(default_factory=list)

class RiskEvent(BaseModel):
    name: str
    p0: confloat(ge=0.0, le=1.0)
    p1: confloat(ge=0.0, le=1.0)
    L0: confloat(ge=0.0)
    L1: confloat(ge=0.0)

class Initiative(BaseModel):
    name: str
    months_accel: confloat(ge=0.0)
    monthly_profit: float
    prob: confloat(ge=0.0, le=1.0)

class OptionOpportunity(BaseModel):
    name: str
    prob: confloat(ge=0.0, le=1.0)
    npv_if_pursued: float
    feasibility_lift: confloat(ge=0.0, le=1.0)
    exercise_cost_reduction_pv: confloat(ge=0.0) = 0.0

class V4OQI(BaseModel):
    flexibility: confloat(ge=0.0, le=5.0) = 0.0
    portability: confloat(ge=0.0, le=5.0) = 0.0
    data_liquidity: confloat(ge=0.0, le=5.0) = 0.0
    scalability: confloat(ge=0.0, le=5.0) = 0.0

class ResilienceScenario(BaseModel):
    name: str
    p: confloat(ge=0.0, le=1.0)
    mttr0_hours: confloat(ge=0.0)
    mttr1_hours: confloat(ge=0.0)
    cost_per_hour: confloat(ge=0.0)

class Confidence(BaseModel):
    v1: confloat(ge=0.0, le=1.0) = 0.3
    v2: confloat(ge=0.0, le=1.0) = 0.3
    v3: confloat(ge=0.0, le=1.0) = 0.3
    v4: confloat(ge=0.0, le=1.0) = 0.3
    v5: confloat(ge=0.0, le=1.0) = 0.3

class Deal(BaseModel):
    meta: Meta
    investment: Investment

    v1_capital_productivity: Optional[V1CapitalProductivity] = None
    v2_risk_events: Optional[List[RiskEvent]] = None
    v3_initiatives: Optional[List[Initiative]] = None
    v4_options: Optional[List[OptionOpportunity]] = None
    v4_oqi: Optional[V4OQI] = None
    v5_resilience: Optional[List[ResilienceScenario]] = None

    confidence: Confidence = Field(default_factory=Confidence)
    assumptions_used: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lengths(self) -> "Deal":
        T = self.meta.horizon_years
        if len(self.investment.opex_annual) != T:
            raise ValueError(f"investment.opex_annual must have length {T} (horizon_years)")
        if self.v1_capital_productivity and self.v1_capital_productivity.fcf_benefit_annual is not None:
            if len(self.v1_capital_productivity.fcf_benefit_annual) != T:
                raise ValueError(f"v1_capital_productivity.fcf_benefit_annual must have length {T}")
        return self
