import asyncio
import os
import json
import re
import sys
from typing import List, Optional
from openai import OpenAI
from pathlib import Path

# Setup project paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.models import Cargo_Action, Cargo_FetchState
from server.environment import CargoComplianceEnv

# CONFIGURATION (Optimized for HF Mistral)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api-inference.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or "your-hf-token-here"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "cargo-compliance")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "CargoComplianceEnv")

# LOGGING
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = f'"{error}"' if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ROBUST JSON PARSER
def repair_json(text: str) -> dict:
    """Strips common LLM conversational filler to find the JSON object."""
    # Look for the first { and the last }
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            return json.loads(json_str)
    except:
        pass
    return {}

def get_llm_response(client: OpenAI, sys_p: str, usr_p: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            temperature=0, # Crucial for consistency
            max_tokens=500
        )
        print(f"LLM RAW RESPONSE: {completion.choices[0].message.content}", flush=True)
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

def merge_extraction(base: dict, update: dict) -> dict:
    if base is None:
        base = {}
    if not isinstance(update, dict):
        return base
    for key, val in update.items():
        if val is not None and str(val).strip() != "":
            base[key] = val
    return base

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CargoComplianceEnv()
    
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        session_id, obs = env.create_task()
        
        # --- PHASE 1: EXTRACTION ---
        steps_taken += 1
        sys_p = "You are a logistics parser. Extract the following into a flat JSON: \"qty\", \"category\", \"Destination\", \"Origin\"."
        usr_p = (f"Extract details from: {obs.manifest.get('raw_text', '')}."
                "Rules:\n"
                "1.'category' MUST be exactly one of: [Food, Nuclear, Pharmaceutical, Electronics].\n"
                "2.'qty' MUST include the number and the unit (e.g., '150 units').\n"
                "3.'Destination' and 'Origin' must be the full country names.\n"
                "4. Output: Raw JSON only.")
        
        extraction_data = repair_json(get_llm_response(client, sys_p, usr_p))

        obs = await env.step(session_id, Cargo_Action(
            action_type=Cargo_FetchState.SUBMIT_EXTRACT,
            decision=json.dumps(extraction_data)
        ))
        rewards.append(obs.reward)
        log_step(step=steps_taken, action="extract", reward=obs.reward, done=False, error=None)

        # --- EXTRACTION REPAIR LOOP (AGENT-DRIVEN QUESTIONS) ---
        max_questions = 3
        questions_used = 0
        while not obs.available_laws and questions_used < max_questions:
            sys_p = (
                "You are a logistics extraction agent. Decide whether to ask ONE clarification question "
                "or to submit a corrected extraction. Output ONLY JSON with this schema:\n"
                "{ \"action\": \"ask\"|\"submit\", \"question\": \"...\", \"extraction\": {\"qty\":...,\"category\":...,\"Destination\":...,\"Origin\":...} }\n"
                "Rules:\n"
                "1) If unsure or the feedback indicates a mismatch, ask a short, specific question about the most uncertain field.\n"
                "2) If confident, set action=submit and provide all four fields.\n"
                "3) Do not include any extra keys or prose."
            )
            usr_p = (
                f"Shipment text: {obs.manifest.get('raw_text', '')}\n"
                f"Current extraction: {json.dumps(extraction_data)}\n"
                f"Env feedback: {obs.text}\n"
                f"Questions used: {questions_used} of {max_questions}\n"
            )
            decision = repair_json(get_llm_response(client, sys_p, usr_p))
            action = (decision.get("action") or "submit").strip().lower()

            if action == "ask" and questions_used < max_questions:
                question = decision.get("question") or "Please confirm the origin, destination, category, and quantity."

                steps_taken += 1
                obs = await env.step(session_id, Cargo_Action(
                    action_type=Cargo_FetchState.FETCH_INFO,
                    decision=question
                ))
                rewards.append(obs.reward)
                log_step(step=steps_taken, action="ask", reward=obs.reward, done=False, error=None)
                questions_used += 1

                # Agent updates extraction from customer reply
                sys_p = (
                    "You are updating a shipment extraction after a customer reply. "
                    "Output ONLY JSON with keys: qty, category, Destination, Origin. "
                    "Preserve existing values if not updated by the reply."
                )
                usr_p = (
                    f"Current extraction: {json.dumps(extraction_data)}\n"
                    f"Customer reply: {obs.text}\n"
                    "Return updated extraction JSON only."
                )
                update = repair_json(get_llm_response(client, sys_p, usr_p))
                extraction_data = merge_extraction(extraction_data, update)

                steps_taken += 1
                obs = await env.step(session_id, Cargo_Action(
                    action_type=Cargo_FetchState.SUBMIT_EXTRACT,
                    decision=json.dumps(extraction_data)
                ))
                rewards.append(obs.reward)
                log_step(step=steps_taken, action="extract_retry", reward=obs.reward, done=False, error=None)
            else:
                update = decision.get("extraction") if isinstance(decision, dict) else None
                extraction_data = merge_extraction(extraction_data, update or {})

                steps_taken += 1
                obs = await env.step(session_id, Cargo_Action(
                    action_type=Cargo_FetchState.SUBMIT_EXTRACT,
                    decision=json.dumps(extraction_data)
                ))
                rewards.append(obs.reward)
                log_step(step=steps_taken, action="extract_retry", reward=obs.reward, done=False, error=None)

            if "answered enough questions" in (obs.text or "").lower():
                break

        # --- PHASE 2: BILATERAL LAW SELECTION ---
        if obs.available_laws:
            steps_taken += 1

            sys_p = (
            "You are a Customs Broker. You must output ONLY a JSON object. "
            "STRICT SCHEMA REQUIRED: You must use these exact keys: 'laws', 'regulator', 'documents'. "
            "STRICT RULE: You may ONLY select Law Names that appear in the 'Available Laws' list provided. Do not use your internal knowledge to suggest extra laws. If a law is not in the list, ignore it."
            "Do NOT use 'Export_law' or 'Import_law' as keys. Put those names inside the 'laws' list."
            "Do NOT use markdown backticks. Return ONLY the raw braces { }"
            )

            usr_p = (
                f"Shipment: {json.dumps(obs.current_extraction)}\n"
                f"Available Laws: {json.dumps(obs.available_laws)}\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. For the 'laws' key, you MUST provide a list of IDs (e.g., ['LAW_001', 'LAW_005']).\n"
                "2. You MUST provide a single string for 'regulator' (e.g., 'CFIA and EU Commission').\n"
                "3. You MUST include essential documents (e.g., ['Health Certificate', 'Invoice'])."
            )
            
            pkg = repair_json(get_llm_response(client, sys_p, usr_p))
            
            # Map IDs to Names
            id_map = {l["id"]: l["name"] for l in obs.available_laws}
            valid_laws = [id_map[lid] for lid in pkg.get("laws", []) if lid in id_map]
            pkg["laws"] = valid_laws

            obs = await env.step(session_id, Cargo_Action(
                action_type=Cargo_FetchState.PICK_LAW,
                decision=json.dumps(pkg)
            ))
            rewards.append(obs.reward)
            log_step(step=steps_taken, action="laws", reward=obs.reward, done=False, error=None)

            # --- PHASE 3: FINAL AUDIT ---
            steps_taken += 1
            sys_p = "Provide 3 sentences of legal reasoning for this shipment."
            usr_p = (f"Compliance Package: {json.dumps(pkg)}"
                    "1. Explicitly mention the Laws and Regulators selected in Phase 2.\n"
                    "2. Explain how the documents provided satisfy the specific Export rules of [Origin] and Import rules of [Destination].\n"
                    "3. Use professional, cite-heavy terminology (e.g., 'Pursuant to...', 'In compliance with...').\n")
            
            reasoning = get_llm_response(client, sys_p, usr_p)
            
            obs = await env.step(session_id, Cargo_Action(
                action_type=Cargo_FetchState.FINAL_VERDICT,
                decision=reasoning
            ))
            rewards.append(obs.reward)
            
            # Success logic
            success = True if sum(rewards) >= 1.5 else False
            log_step(step=steps_taken, action="audit", reward=obs.reward, done=True, error=None)

    except Exception as e:
        log_step(step=steps_taken + 1, action="fail", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
