import asyncio
import os
import json
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

# --- MANDATORY HACKATHON CONFIGURATION ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "cargo-compliance")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "CargoComplianceEnv")

# Define max possible reward to normalize score to [0, 1]
# Based on your logs: Extract(0.80) + Laws(1.67) + Audit(1.80) = ~4.27
MAX_TOTAL_REWARD = 5.5 
SUCCESS_SCORE_THRESHOLD = 0.44 # ~2.0 / 4.5

# --- MANDATORY STDOUT LOGGERS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = f'"{error}"' if error else "null"
    done_val = str(done).lower()
    # Note: action string might have spaces, we replace newlines to keep it on one line
    safe_action = str(action).replace('\n', ' ')
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# --- HELPER FUNCTIONS ---
def repair_json(text: str) -> dict:
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
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
            temperature=0, 
            max_tokens=500
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

def merge_extraction(base: dict, update: dict) -> dict:
    if not isinstance(update, dict):
        return base or {}
    for key, val in update.items():
        if val and str(val).strip() != "":
            base[key] = val
    return base


# --- MAIN EXECUTION LOOP ---
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize your local environment
    env = CargoComplianceEnv()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        session_id, obs = env.create_task()
        
        # --- PHASE 1: EXTRACTION ---
        steps_taken += 1
        sys_p = "You are a logistics parser. Extract into flat JSON: \"qty\", \"category\", \"Destination\", \"Origin\"."
        usr_p = f"Extract details from: {obs.manifest.get('raw_text', '')}. Raw JSON only."
        
        extraction_data = repair_json(get_llm_response(client, sys_p, usr_p))

        obs = await env.step(session_id, Cargo_Action(
            action_type=Cargo_FetchState.SUBMIT_EXTRACT,
            decision=json.dumps(extraction_data)
        ))
        
        reward = obs.reward or 0.0
        rewards.append(reward)
        log_step(step=steps_taken, action="extract", reward=reward, done=False, error=None)

        # --- EXTRACTION REPAIR LOOP ---
        questions_used = 0
        max_extract_retries = 3
        extract_retries = 0

        while not obs.available_laws and questions_used < 3 and extract_retries < max_extract_retries:
            extract_retries += 1 # Prevents infinite re-submissions
            
            sys_p = "Output ONLY JSON: { \"action\": \"ask\"|\"submit\", \"question\": \"...\", \"extraction\": {...} }"
            usr_p = f"Current extraction: {json.dumps(extraction_data)}\nEnv feedback: {obs.text}"
            
            decision = repair_json(get_llm_response(client, sys_p, usr_p))
            action_type = (decision.get("action") or "submit").strip().lower()

            if action_type == "ask":
                question = decision.get("question", "Confirm details.")
                steps_taken += 1
                
                obs = await env.step(session_id, Cargo_Action(action_type=Cargo_FetchState.FETCH_INFO, decision=question))
                rewards.append(obs.reward)
                log_step(step=steps_taken, action="ask", reward=obs.reward, done=False, error=None)
                questions_used += 1

                update = repair_json(get_llm_response(client, "Update extraction JSON.", f"Reply: {obs.text}"))
                extraction_data = merge_extraction(extraction_data, update)
                
                steps_taken += 1
                obs = await env.step(session_id, Cargo_Action(action_type=Cargo_FetchState.SUBMIT_EXTRACT, decision=json.dumps(extraction_data)))
                rewards.append(obs.reward)
                log_step(step=steps_taken, action="extract_retry", reward=obs.reward, done=False, error=None)
            else:
                update = decision.get("extraction") if isinstance(decision, dict) else None
                extraction_data = merge_extraction(extraction_data, update or {})
                
                steps_taken += 1
                obs = await env.step(session_id, Cargo_Action(action_type=Cargo_FetchState.SUBMIT_EXTRACT, decision=json.dumps(extraction_data)))
                rewards.append(obs.reward)
                log_step(step=steps_taken, action="extract_retry", reward=obs.reward, done=False, error=None)

            if "answered enough questions" in (obs.text or "").lower():
                break

        # --- PHASE 2 & 3: LAW SELECTION & AUDIT ---
        if obs.available_laws:
            steps_taken += 1
            sys_p = "Output ONLY JSON schema: 'laws', 'regulator', 'documents'. Pick laws ONLY from the provided list."
            usr_p = f"Shipment: {json.dumps(obs.current_extraction)}\nAvailable: {json.dumps(obs.available_laws)}"
            
            pkg = repair_json(get_llm_response(client, sys_p, usr_p))
            
            # Map IDs to Names securely
            id_map = {l["id"]: l["name"] for l in obs.available_laws}
            pkg["laws"] = [id_map[lid] for lid in pkg.get("laws", []) if lid in id_map]

            obs = await env.step(session_id, Cargo_Action(action_type=Cargo_FetchState.PICK_LAW, decision=json.dumps(pkg)))
            rewards.append(obs.reward)
            log_step(step=steps_taken, action="laws", reward=obs.reward, done=False, error=None)

            # Final Audit
            steps_taken += 1
            reasoning = get_llm_response(client, "Provide 3 sentences of legal reasoning.", f"Package: {json.dumps(pkg)}")
            obs = await env.step(session_id, Cargo_Action(action_type=Cargo_FetchState.FINAL_VERDICT, decision=reasoning))
            rewards.append(obs.reward)
            log_step(step=steps_taken, action="audit", reward=obs.reward, done=True, error=None)

        # Normalize score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0) 
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken + 1, action="fail", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())