import json
import streamlit as st

from eve_models import Deal
from eve_scoring import compute_eve, EVEConfig
from llm_intake import ask_intake_agent

st.set_page_config(page_title="EVE Scoring Engine", layout="wide")
st.title("Enterprise Value Engineering™ (EVE) — Scoring Engine")

# --- Secrets / API Key ---
OPENAI_API_KEY = (st.secrets.get("OPENAI_API_KEY", "") or "").strip()
OPENAI_MODEL = (st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini") or "").strip()

with st.sidebar:
    st.markdown("### LLM Status")
    st.write("API key loaded:", "✅" if len(OPENAI_API_KEY) > 20 else "❌")
    st.write("Model:", OPENAI_MODEL)
    if len(OPENAI_API_KEY) <= 20:
        st.info("Set OPENAI_API_KEY in Streamlit Cloud → App → Manage app → Settings → Secrets.")

tab1, tab2 = st.tabs(["LLM Intake (Chat → Score)", "JSON Scorer"])

# -------------------------
# TAB 1: LLM Intake
# -------------------------
with tab1:
    st.subheader("LLM Intake → Validated Deal JSON → EVI")

    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Cloud Secrets or .streamlit/secrets.toml locally.")
        st.stop()

    if "intake_messages" not in st.session_state:
        st.session_state.intake_messages = []
    if "deal_json" not in st.session_state:
        st.session_state.deal_json = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = "Let’s start. What’s the company’s industry, approximate annual revenue, and EBITDA margin (or ‘unknown’)?"

    # Display chat history
    for m in st.session_state.intake_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # If we haven't started, show initial prompt
    if len(st.session_state.intake_messages) == 0:
        with st.chat_message("assistant"):
            st.write(st.session_state.last_question)

    user_input = st.chat_input("Type your answer…")
    if user_input:
        st.session_state.intake_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Call LLM intake agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                out = ask_intake_agent(
                    api_key=OPENAI_API_KEY,
                    messages=st.session_state.intake_messages,
                    model=OPENAI_MODEL,
                )

            if out["type"] == "question":
                st.write(out["text"])
                st.session_state.intake_messages.append({"role": "assistant", "content": out["text"]})
                st.session_state.last_question = out["text"]

            elif out["type"] == "deal_json":
                st.success("✅ Deal JSON created and validated.")
                st.session_state.deal_json = out["deal"]

                # Score it
                deal = Deal.model_validate(st.session_state.deal_json)
                config = EVEConfig()
                result = compute_eve(deal, config=config, run_sensitivity=True)

                st.markdown(
                    f"### Results\n"
                    f"- **EVI:** {result['EVI']:.1f}\n"
                    f"- **EVI (confidence-adjusted):** {result['EVI_conf']:.1f}\n"
                    f"- **Weighted confidence:** {result['confidence_weighted']:.2f}"
                )

                if result["warnings"]:
                    st.warning("\n".join(result["warnings"]))

                colA, colB = st.columns(2)
                with colA:
                    st.subheader("Pillar Scores")
                    st.json(result["pillar_scores"])
                with colB:
                    st.subheader("PV Benefits")
                    st.json(result["pillar_pv_benefits"])

                st.subheader("Sensitivity (ΔEVI from +10% bump)")
                st.json(result["sensitivities"])

                st.subheader("Generated Deal JSON")
                st.code(json.dumps(st.session_state.deal_json, indent=2), language="json")

    st.divider()
    colx, coly = st.columns([1, 1])
    with colx:
        if st.button("Reset intake chat"):
            st.session_state.intake_messages = []
            st.session_state.deal_json = None
            st.session_state.last_question = "Let’s start. What’s the company’s industry, approximate annual revenue, and EBITDA margin (or ‘unknown’)?"
            st.rerun()
    with coly:
        st.caption("Tip: Keep answers short. If you don’t know a number, say 'unknown' and the agent will use conservative assumptions.")

# -------------------------
# TAB 2: JSON Scorer (your existing mode)
# -------------------------
with tab2:
    st.subheader("Paste Deal JSON → Compute EVI")

    default_json = {
        "meta": {"company": {"industry": "manufacturing", "revenue": 500000000, "ebitda_margin": 0.18},
                 "horizon_years": 5, "discount_rate": 0.10, "currency": "USD"},
        "investment": {"capex_upfront": 12000000, "opex_annual": [1500000]*5},
        "v1_capital_productivity": {"fcf_benefit_annual": [3000000]*5,
                                    "notes": [{"text": "Maintenance + labor + refresh deferral", "source": "estimated"}]},
        "v2_risk_events": [
            {"name": "ransomware", "p0": 0.12, "p1": 0.05, "L0": 15000000, "L1": 12000000},
            {"name": "major_outage", "p0": 0.15, "p1": 0.08, "L0": 6000000, "L1": 4500000}
        ],
        "v3_initiatives": [{"name": "digital_product_launch", "months_accel": 4, "monthly_profit": 800000, "prob": 0.7}],
        "v4_options": [{"name": "ai_product_line", "prob": 0.4, "npv_if_pursued": 20000000, "feasibility_lift": 0.3, "exercise_cost_reduction_pv": 0}],
        "v4_oqi": {"flexibility": 4, "portability": 4, "data_liquidity": 3, "scalability": 4},
        "v5_resilience": [{"name": "major_outage", "p": 0.15, "mttr0_hours": 40, "mttr1_hours": 15, "cost_per_hour": 250000}],
        "confidence": {"v1": 0.7, "v2": 0.6, "v3": 0.5, "v4": 0.4, "v5": 0.6},
        "assumptions_used": []
    }

    txt = st.text_area("Deal JSON", value=json.dumps(default_json, indent=2), height=380)

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

    if st.button("Compute EVI (JSON)"):
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

            st.subheader("Outputs")
            st.json(result)

        except Exception as e:
            st.exception(e)
