from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import ValidationError

from eve_models import Deal


INTAKE_INSTRUCTIONS = """
You are EVE Intake Agent for Enterprise Value Engineering™.

Goal:
Collect information needed to compute the EVE score (EVI) for an IT hardware/software investment.
You must output a single JSON object that conforms to the EVE schema contract.

Behavior:
- Ask follow-up questions ONLY when required fields are missing or logically inconsistent.
- Minimize questions: prefer conservative defaults if user does not know, but MUST mark those values as "assumed" in notes and reduce pillar confidence accordingly.
- Never double count: do not place the same loss reduction in both V2 and V5.
- Track confidence per pillar (0.0–1.0) based on evidence quality:
  0.9–1.0 measured/audited, 0.6–0.8 strong internal estimate, 0.3–0.5 directional/vendor benchmark, 0.1–0.2 speculative
- Units:
  USD for money, probabilities decimals [0,1], MTTR in hours, monthly_profit in USD/month.
- If user provides ranges, store central estimate and include range in notes.

Conversation flow (script):
1) Ask: industry, revenue, EBITDA margin (or unknown)
2) Ask: horizon (default 5) and discount rate (default 10%)
3) Ask: upfront capex + annual opex (or year-by-year)
4) Ask: choose 2–3 pillars (Productivity, Risk, Velocity, Optionality, Resilience)
5) Ask only the relevant module questions for chosen pillars (2 questions each)
6) Run sanity check for double counting, units, missing required arrays length = horizon

Output requirement:
When you have enough information to produce a valid Deal JSON, output ONLY JSON.
No markdown. No explanations. Just JSON.
Include assumptions_used as a list. Include notes under pillars with text + source ("provided"|"assumed"|"estimated").
If you still need information, ask ONE concise question at a time.
""".strip()


def _openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction if the model returns JSON text.
    (We still validate with Pydantic.)
    """
    text = text.strip()
    if not text:
        return None
    # Fast path: starts with JSON object
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return None
    # Try to find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def ask_intake_agent(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-5",
) -> Dict[str, Any]:
    """
    messages: list of {"role": "user"|"assistant", "content": "..."} from the app chat.
    Returns:
      {"type": "question", "text": "..."} OR {"type": "deal_json", "deal": {...}}
    """
    client = _openai_client(api_key)

    # Convert chat history into a single input string with roles.
    # (Keeps it simple; you can switch to richer message inputs later.)
    transcript_lines = []
    for m in messages:
        role = m["role"].upper()
        transcript_lines.append(f"{role}: {m['content']}")
    transcript = "\n".join(transcript_lines).strip()

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INTAKE_INSTRUCTIONS},
            {"role": "user", "content": transcript}
        ],
    )

text = (chat.choices[0].message.content or "").strip()

    # If it looks like JSON, try validate as Deal
    maybe = _extract_json_from_text(text)
    if maybe is not None:
        try:
            deal = Deal.model_validate(maybe)
            return {"type": "deal_json", "deal": deal.model_dump()}
        except ValidationError as ve:
            # Ask the model to repair, using the validation errors
            repair_prompt = (
                "Your JSON did not validate against the schema.\n"
                "Fix the JSON to pass validation. Output ONLY corrected JSON.\n"
                f"Validation errors:\n{ve}\n\n"
                f"Bad JSON:\n{text}\n"
            )
            resp2 = client.responses.create(
                model=model,
                instructions=INTAKE_INSTRUCTIONS,
                input=repair_prompt,
            )
            text2 = (resp2.output_text or "").strip()
            maybe2 = _extract_json_from_text(text2)
            if maybe2 is not None:
                deal2 = Deal.model_validate(maybe2)
                return {"type": "deal_json", "deal": deal2.model_dump()}

            return {"type": "question", "text": "I couldn’t produce valid Deal JSON yet. Please answer: what is the upfront CapEx and annual OpEx?"}

    # Otherwise, treat as the next question
    return {"type": "question", "text": text if text else "What industry is the company in, and what’s the approximate annual revenue?"}
