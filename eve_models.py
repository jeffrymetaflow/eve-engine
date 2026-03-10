from __future__ import annotations
from typing import List, Optional, Literal, Annotated
from pydantic import BaseModel, Field, model_validator

# --- Types ---
SourceType = Literal["provided", "assumed", "estimated"]

# --- Components ---
class Note(BaseModel):
    text: str
    source: SourceType = "provided"

class Company(BaseModel):
    industry: str
    revenue: Optional[float] = Field(default=None, ge=0.0)
    ebitda_margin: Optional[float] = Field(default=None, ge=-1.0, le=1.0)

class Meta(BaseModel):
    company: Company
    horizon_years: Annotated[int, Field(ge=1, le=15)] = 5
    discount_rate: Annotated[float, Field(ge=0.0, le=0.5)] = 0.10
    currency: str = "USD"

class Investment(BaseModel):
    capex_upfront: float = Field(default=0.0, ge=0.0)
    opex_annual: List[float] = Field(default_factory=list)

class V1CapitalProductivity(BaseModel):
    fcf_benefit_annual: Optional[List[float]] = None
    notes: List[Note] = Field(default_factory=list)

class RiskEvent(BaseModel):
    name: str
    p0: float = Field(ge=0.0, le=1.0)
    p1: float = Field(ge=0.0, le=1.0)
    L0: float = Field(ge=0.0)
    L1: float = Field(ge=0.0)

class Initiative(BaseModel):
    name: str
    months_accel: float = Field(ge=0.0)
    monthly_profit: float
    prob: float = Field(ge=0.0, le=1.0)

class OptionOpportunity(BaseModel):
    name: str
    prob: float = Field(ge=0.0, le=1.0)
    npv_if_pursued: float
    feasibility_lift: float = Field(ge=0.0, le=1.0)
    exercise_cost_reduction_pv: float = Field(default=0.0, ge=0.0)

class V4OQI(BaseModel):
    flexibility: float = Field(default=0.0, ge=0.0, le=5.0)
    portability: float = Field(default=0.0, ge=0.0, le=5.0)
    data_liquidity: float = Field(default=0.0, ge=0.0, le=5.0)
    scalability: float = Field(default=0.0, ge=0.0, le=5.0)

class ResilienceScenario(BaseModel):
    name: str
    p: float = Field(ge=0.0, le=1.0)
    mttr0_hours: float = Field(ge=0.0)
    mttr1_hours: float = Field(ge=0.0)
    cost_per_hour: float = Field(ge=0.0)

class Confidence(BaseModel):
    v1: float = Field(default=0.3, ge=0.0, le=1.0)
    v2: float = Field(default=0.3, ge=0.0, le=1.0)
    v3: float = Field(default=0.3, ge=0.0, le=1.0)
    v4: float = Field(default=0.3, ge=0.0, le=1.0)
    v5: float = Field(default=0.3, ge=0.0, le=1.0)

# --- Main Deal Model ---
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
    def validate_and_fix_lengths(self) -> "Deal":
        T = self.meta.horizon_years
        
        # 1. Fix Investment OpEx
        current_opex = self.investment.opex_annual
        if not current_opex:
            self.investment.opex_annual = [0.0] * T
        elif len(current_opex) < T:
            # Pad with the last value provided
            last_val = current_opex[-1]
            padding = [last_val] * (T - len(current_opex))
            self.investment.opex_annual = current_opex + padding
        elif len(current_opex) > T:
            self.investment.opex_annual = current_opex[:T]

        # 2. Fix V1 Capital Productivity Benefit
        v1 = self.v1_capital_productivity
        if v1 and v1.fcf_benefit_annual is not None:
            current_fcf = v1.fcf_benefit_annual
            if len(current_fcf) < T:
                last_val = current_fcf[-1] if current_fcf else 0.0
                padding = [last_val] * (T - len(current_fcf))
                v1.fcf_benefit_annual = current_fcf + padding
            elif len(current_fcf) > T:
                v1.fcf_benefit_annual = current_fcf[:T]
        
        return self
