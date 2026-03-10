from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import ValidationError
from eve_models import Deal

INTAKE_INSTRUCTIONS = """
You are the EVE Intake Agent for Enterprise Value Engineering™.
Goal: Collect info for IT investment scoring.
Output: If you have enough info, output ONLY the raw JSON. Otherwise, ask ONE concise question.
...
(Rest of your instructions remain excellent)
""".strip()

def _openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text: return None
    # Look for the JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except:
            return None
    return None

def ask_intake_agent(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo-0125", # Use a valid model name
) -> Dict[str, Any]:
    client = _openai_client(api_key)

    # 1. Native Message Passing (Better for LLM performance)
    openai_messages = [{"role": "system", "content": INTAKE_INSTRUCTIONS}]
    for m in messages:
        openai_messages.append({"role": m["role"], "content": m["content"]})

    # 2. Initial Call
    completion = client.chat.completions.create(
        model=model,
        messages=openai_messages,
        temperature=0.2, # Lower temperature = more stable JSON
    )
    
    text = (completion.choices[0].message.content or "").strip()
    maybe = _extract_json_from_text(text)

    # 3. Validation and Repair Logic
    if maybe is not None:
        try:
            deal = Deal.model_validate(maybe)
            return {"type": "deal_json", "deal": deal.model_dump()}
        except ValidationError as ve:
            # Repair Attempt
            repair_messages = openai_messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"The JSON failed validation:\n{ve}\nFix it and output ONLY valid JSON."}
            ]
            repair_completion = client.chat.completions.create(
                model=model,
                messages=repair_messages
            )
            text2 = (repair_completion.choices[0].message.content or "").strip()
            maybe2 = _extract_json_from_text(text2)
            if maybe2:
                try:
                    deal2 = Deal.model_validate(maybe2)
                    return {"type": "deal_json", "deal": deal2.model_dump()}
                except:
                    pass
            
            return {"type": "question", "text": "I had trouble formatting the deal data. Can you clarify the annual OpEx values?"}

    # 4. If no JSON, treat as a conversation question
    return {"type": "question", "text": text if text else "What industry is the company in?"}
