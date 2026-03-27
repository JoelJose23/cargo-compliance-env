import random
import uuid
import json
import os
from typing import Dict, Any, Tuple
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from openenv.core.env_server import Environment
from models import Cargo_Action, Cargo_Observation, Cargo_FetchState, Cargo_State

# --- Initialize Groq Client ---
# Ensure you have 'GROQ_API_KEY' set in your environment variables/Docker secrets
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

AVAILABLE_LAWS = [
    {"id": "AGRI_001", "name": "USDA Perishable Import Law", "category": "Agriculture"},
    {"id": "BIO_SAFETY_02", "name": "Bio-Hazard Containment Act", "category": "Agriculture"},
    {"id": "FLOWER_REG_01", "name": "CITES Floral Export Protocol", "category": "Flora"},
    {"id": "CHEM_DUAL_01", "name": "Dual-Use Chemical Export Treaty", "category": "Chemical"},
    {"id": "HAZMAT_TRANS_04", "name": "Hazardous Material Transport Act", "category": "Chemical"},
    {"id": "TEXTILE_PEST_01", "name": "Vintage Textile Pest Control", "category": "Textile"},
    {"id": "ELECTRONIC_WASTE_04", "name": "E-Waste Disposal Mandate", "category": "Electronics"}
]

PROMPT_POOL = [
    {
        "text": "Shipping 200 units of Organic Cavendish from Ecuador to Rotterdam port.",
        "truth": {
            "goods": "Banana", 
            "qty": "200 units", 
            "category": "Agriculture",
            "Destination": "Rotterdam", # Fixed: Was Ecuador
            "Origin": "Ecuador",       # Fixed: Was Cavendish
            "required_laws": ["AGRI_001", "BIO_SAFETY_02"],
            "red_herrings": ["FLOWER_REG_01", "ELECTRONIC_WASTE_04"]
        }
    },
    {
        "text": "Emergency transport of 50kg Potassium Nitrate for industrial fertilizer from India to Dubai.",
        "truth": {
            "goods": "Potassium Nitrate", 
            "qty": "50kg", 
            "category": "Chemical",
            "Destination": "Dubai",
            "Origin": "India",
            "required_laws": ["CHEM_DUAL_01", "HAZMAT_TRANS_04"],
            "red_herrings": ["AGRI_001", "TEXTILE_PEST_01"]
        }
    }
]

class CargoComplianceEnv(Environment):
    def __init__(self):
        self.sessions: Dict[str, Cargo_State] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_llm_judge_score(self, reasoning: str, extraction: dict, truth: dict) -> float:
        """Calls Groq to audit the reasoning. Uses retry logic to prevent rate-limit crashes."""
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a customs auditor. Grade the agent's reasoning from 0.0 to 1.0 based on how well they justify their selected laws against the shipping context. Respond ONLY with the float number."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {json.dumps(extraction)}\nAgent Reasoning: {reasoning}\nCorrect Laws: {truth['required_laws']}"
                    }
                ],
                model="llama-3.1-8b-instant", # High-speed, lower token cost
                max_tokens=10,
                temperature=0,
            )
            return float(chat_completion.choices[0].message.content.strip())
        except Exception:
            return 0.5 # Neutral fallback if API fails after retries

    def create_task(self) -> Tuple[str, Cargo_Observation]:
        session_id = str(uuid.uuid4())
        selected_task = random.choice(PROMPT_POOL)
        
        state = Cargo_State(
            task_id=session_id,
            steps=0,
            history=[],
            phase="EXTRACTION",
            questions_asked=0,
            extraction_data={"goods": None, "qty": None, "category": None, "Destination": None, "Origin": None, "laws": []}
        )
        state.ground_truth = selected_task["truth"]
        self.sessions[session_id] = state
        
        initial_obs = Cargo_Observation(
            text=f"NEW SHIPMENT: {selected_task['text']}\nExtract into JSON: goods, qty, category, Destination, Origin. Max 3 questions. Penalty: -0.1/question, -1.0/wrong guess.",
            current_extraction=state.extraction_data,
            available_laws=[],
            manifest={"raw_text": selected_task["text"]},
            laws=[],
            history=[],
            step=0,
            reward=0.0
        )
        return session_id, initial_obs

    async def step(self, session_id: str, action: Cargo_Action) -> Cargo_Observation:
        state = self.sessions.get(session_id)
        if not state: raise ValueError("Session not found.")

        truth = state.ground_truth
        reward = 0.0
        obs_text = ""
        is_terminal = False
        state.steps += 1

        # --- PHASE 1: EXTRACTION ---
        if state.phase == "EXTRACTION":
            if action.action_type == Cargo_FetchState.FETCH_INFO:
                if state.questions_asked < 3:
                    state.questions_asked += 1
                    reward = -0.1
                    obs_text = f"Clarification provided. ({state.questions_asked}/3)"
                else:
                    obs_text = "Question limit reached. Submit JSON."
            
            elif action.action_type == Cargo_FetchState.VERDICT:
                try:
                    data = json.loads(action.decision)
                    # Fixed: Added 'qty' validation and Destination/Origin checks
                    if (data.get("goods") == truth["goods"] and 
                        data.get("qty") == truth["qty"] and 
                        data.get("category") == truth["category"] and
                        data.get("Destination") == truth["Destination"] and
                        data.get("Origin") == truth["Origin"]):
                        
                        reward = 1.0 - (state.questions_asked * 0.1)
                        state.phase = "SELECTION"
                        state.extraction_data.update(data)
                        obs_text = "Extraction Verified. Phase 2: Select Laws."
                    else:
                        reward = -1.0
                        obs_text = "Extraction Failed: Mismatch in data."
                except json.JSONDecodeError:
                    reward = -1.0
                    obs_text = "ERROR: Invalid JSON format."

        # --- PHASE 2: LAW SELECTION ---
        elif state.phase == "SELECTION":
            if action.action_type == Cargo_FetchState.PICK_LAW:
                try:
                    selected_laws = json.loads(action.decision)
                    state.extraction_data["laws"] = selected_laws
                    for law in selected_laws:
                        if law in truth["required_laws"]: reward += 1.0
                        elif law in truth["red_herrings"]: reward -= 1.5
                        else: reward -= 0.5
                    
                    state.phase = "VERDICT"
                    obs_text = "Laws selected. Final Step: Submit Reasoning."
                except json.JSONDecodeError:
                    reward = -1.0
                    obs_text = "ERROR: List laws in JSON format."

        # --- PHASE 3: FINAL AUDIT (THE JUDGE) ---
        elif state.phase == "VERDICT":
            if action.action_type == Cargo_FetchState.VERDICT:
                # Call the Groq Judge
                audit_score = await self.get_llm_judge_score(action.decision, state.extraction_data, truth)
                reward = audit_score * 2.0 # Weight the final logic score
                obs_text = f"Audit Complete. Judge Score: {audit_score}. Episode Finished."
                is_terminal = True

        return Cargo_Observation(
            text=obs_text,
            current_extraction=state.extraction_data,
            available_laws=AVAILABLE_LAWS if state.phase == "SELECTION" else [],
            manifest={},
            laws=state.extraction_data.get("laws", []),
            history=state.history,
            step=state.steps,
            reward=reward,
            is_terminal=is_terminal
        )