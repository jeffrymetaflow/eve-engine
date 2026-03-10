from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import ValidationError
from eve_models import Deal
from json_repair import repair_json  # Ensure you 'pip install json-repair'

# STRENGTHENED INSTRUCTIONS: Explicitly handles the OpEx math issues
INTAKE_INSTRUCTIONS = """
You are the EVE Intake Agent for Enterprise Value Engineering™.
Goal: Collect info for IT investment scoring.

### FINANCIAL RULES:
- Extract all monetary values as INTEGERS (e.g., 5000, not "$5,000").
- If a value is given as Monthly, multiply by 12 to get the ANNUAL value.
- If a value is given as a range (e.g., 10k-20k), use the midpoint (15000).

### OUTPUT RULES:
- If you have enough info to fill the 'Deal' model, output ONLY the raw JSON.
- If info is missing, ask ONE concise question to get the missing piece.
- DO NOT add conversational filler when outputting JSON.
""".strip()

def _openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Uses json-repair to handle GPT-3.5's formatting quirks."""
    text = text.strip()
    if not text: return None
    
    # Locate potential JSON boundaries
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1:
        json_str = text[start : end + 1]
        try:
            # repair_json handles missing quotes, trailing commas, etc.
            repaired = repair_json(json_str)
            return json.loads(repaired)
        except:
            return None
    return None

def ask_intake_agent(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo", 
) -> Dict[str, Any]:
    client = _openai_client(api_key)

    # 1. Setup Messages
    openai_messages = [{"role": "system", "content": INTAKE_INSTRUCTIONS}]
    for m in messages:
        openai_messages.append({"role": m["role"], "content": m["content"]})

    # 2. Call LLM with JSON-Mode enabled (if supported)
    # Note: gpt-3.5-turbo supports response_format in newer versions
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=0, # 0 is better for extraction than 0.2
            response_format={ "type": "json_object" } if "turbo" in model else None
        )
    except Exception as e:
        # Fallback if the specific model version doesn't support JSON mode
        completion = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=0
        )
    
    text = (completion.choices[0].message.content or "").strip()
    maybe = _safe_parse_json(text)

    # 3. Validation and Smart Repair Logic
    if maybe is not None:
        try:
            deal = Deal.model_validate(maybe)
            return {"type": "deal_json", "deal": deal.model_dump()}
        except ValidationError as ve:
            # Instead of just saying it failed, we send the error BACK to the LLM
            repair_messages = openai_messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Your JSON had these errors: {ve}. Please correct the formatting and units."}
            ]
            repair_completion = client.chat.completions.create(
                model=model,
                messages=repair_messages,
                temperature=0
            )
            text2 = (repair_completion.choices[0].message.content or "").strip()
            maybe2 = _safe_parse_json(text2)
            
            if maybe2:
                try:
                    deal2 = Deal.model_validate(maybe2)
                    return {"type": "deal_json", "deal": deal2.model_dump()}
                except:
                    pass
            
            return {"type": "question", "text": "I'm having a little trouble calculating the annual OpEx. Could you provide those costs in a simple yearly total?"}

    # 4. Fallback to conversation
    return {"type": "question", "text": text if text else "Could you tell me a bit more about the deal's operational costs?"}
