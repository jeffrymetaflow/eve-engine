from __future__ import annotations
from typing import List, Optional, Literal, Annotated
from pydantic import BaseModel, Field, conint, confloat, model_validator

SourceType = Literal["provided", "assumed", "estimated"]

class Note(BaseModel):
    text: str
    source: SourceType = "provided"

class Company(BaseModel):
    industry: str
    # Changed to default to 0.0 or handle None more gracefully in scoring
    revenue: Optional[float] = Field(default=None, ge=0.0)
    ebitda_margin: Optional[float] = Field(default=None, ge=-1.0, le=1.0)

class Meta(BaseModel):
    company: Company
    horizon_years: Annotated[int, Field(ge=1, le=15)] = 5
    discount_rate: Annotated[float, Field(ge=0.0, le=0.5)] = 0.10
    currency: str = "USD"

class Investment(BaseModel):
    capex_upfront: float = Field(default=0.0, ge=0.0)
    # Default factory to avoid validation errors before the LLM finishes
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
        
        # Check OPEX length
        if len(self.investment.opex_annual) != T:
            # Instead of failing, we could pad with 0s, but raising Error is safer for financial models
            raise ValueError(f"investment.opex_annual must have length {T} to match horizon_years.")
        
        # Check V1 Benefit length
        if self.v1_capital_productivity and self.v1_capital_productivity.fcf_benefit_annual is not None:
            if len(self.v1_capital_productivity.fcf_benefit_annual) != T:
                raise ValueError(f"v1_capital_productivity.fcf_benefit_annual must have length {T}.")
        
        return self
