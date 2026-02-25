import json
import streamlit as st

from eve_models import Deal
from eve_scoring import compute_eve, EVEConfig

st.set_page_config(page_title="EVE Scoring Engine", layout="wide")
st.title("Enterprise Value Engineering™ (EVE) — Scoring Engine")

st.write("Paste a deal JSON and compute EVI. (LLM intake will generate this JSON later.)")

default_json = {
  "meta": {"company": {"industry":"manufacturing","revenue":500000000,"ebitda_margin":0.18},
           "horizon_years":5,"discount_rate":0.10,"currency":"USD"},
  "investment": {"capex_upfront":12000000,"opex_annual":[1500000,1500000,1500000,1500000,1500000]},
  "v1_capital_productivity": {"fcf_benefit_annual":[3000000,3000000,3000000,3000000,3000000],
                              "notes":[{"text":"Maintenance + labor + refresh deferral","source":"estimated"}]},
  "v2_risk_events": [
    {"name":"ransomware","p0":0.12,"p1":0.05,"L0":15000000,"L1":12000000},
    {"name":"major_outage","p0":0.15,"p1":0.08,"L0":6000000,"L1":4500000}
  ],
  "v3_initiatives": [
    {"name":"digital_product_launch","months_accel":4,"monthly_profit":800000,"prob":0.7}
  ],
  "v4_options": [
    {"name":"ai_product_line","prob":0.4,"npv_if_pursued":20000000,"feasibility_lift":0.3,"exercise_cost_reduction_pv":0}
  ],
  "v4_oqi": {"flexibility":4,"portability":4,"data_liquidity":3,"scalability":4},
  "v5_resilience": [
    {"name":"major_outage","p":0.15,"mttr0_hours":40,"mttr1_hours":15,"cost_per_hour":250000}
  ],
  "confidence": {"v1":0.7,"v2":0.6,"v3":0.5,"v4":0.4,"v5":0.6},
  "assumptions_used": []
}

txt = st.text_area("Deal JSON", value=json.dumps(default_json, indent=2), height=420)

st.subheader("Scoring Parameters")
col1, col2 = st.columns([1, 1])
with col1:
    logistic_a = st.number_input("Logistic steepness (a)", value=6.0, step=0.5)
    logistic_b = st.number_input("Logistic midpoint (b)", value=0.10, step=0.05)
with col2:
    st.write("Weights (must sum to 1.00)")
    w1 = st.slider("V1 Capital Productivity", 0.0, 1.0, 0.25, 0.01)
    w2 = st.slider("V2 Risk Compression", 0.0, 1.0, 0.20, 0.01)
    w3 = st.slider("V3 Strategic Velocity", 0.0, 1.0, 0.20, 0.01)
    w4 = st.slider("V4 Optionality", 0.0, 1.0, 0.20, 0.01)
    w5 = st.slider("V5 Resilience", 0.0, 1.0, 0.15, 0.01)

if st.button("Compute EVI"):
    try:
        data = json.loads(txt)
        deal = Deal.model_validate(data)

        weights = {"v1": w1, "v2": w2, "v3": w3, "v4": w4, "v5": w5}
        s = sum(weights.values())
        if abs(s - 1.0) > 1e-6:
            st.error(f"Weights must sum to 1.0. Current sum = {s:.3f}")
            st.stop()

        config = EVEConfig(weights=weights, logistic_a=logistic_a, logistic_b=logistic_b)
        result = compute_eve(deal, config=config, run_sensitivity=True)

        st.success(
            f"EVI = {result['EVI']:.1f} | "
            f"EVI (conf adj) = {result['EVI_conf']:.1f} | "
            f"weighted confidence = {result['confidence_weighted']:.2f}"
        )

        if result["warnings"]:
            st.warning("\n".join(result["warnings"]))

        st.subheader("Pillar Summary")
        st.json({
            "pv_cost": result["pv_cost"],
            "pillar_pv_benefits": result["pillar_pv_benefits"],
            "pillar_ratios": result["pillar_ratios"],
            "pillar_scores": result["pillar_scores"],
            "sensitivities": result["sensitivities"]
        })

    except Exception as e:
        st.exception(e)
