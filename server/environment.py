import random
import uuid
import json
import os
from typing import Dict, Any, Tuple, Optional, List
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from openenv.core.env_server import Environment
from models import Cargo_Action, Cargo_Observation, Cargo_FetchState, Cargo_State
import re
from dotenv import load_dotenv

load_dotenv()

# --- Initialize Groq Client ---
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Data Loading Engine ---
def load_environment_data(json_path: str) -> Tuple[List[Dict], List[Dict]]:
    with open(json_path, "r") as f:
        raw_data = json.load(f)
        
    available_laws = []
    prompt_pool = []
    
    category_map = {
        "Food Products (Processed, Packaged, Agricultural)": "Food",
        "Radioactive Materials and Nuclear Goods": "Nuclear",
        "Pharmaceutical Drugs (Finished Formulations, APIs, Controlled Substances)": "Pharmaceutical",
        "Electronics (Consumer, Industrial, and Dual-use Goods)": "Electronics"
    }
    
    # 1. Build the AVAILABLE_LAWS registry
    law_counter = 1
    for industry_block in raw_data:
        category = category_map.get(industry_block["industry"], "General")
        for country in industry_block.get("countries", []):
            for law in country.get("laws", []):
                if not any(l["name"] == law for l in available_laws):
                    available_laws.append({
                        "id": f"LAW_{law_counter:03d}",
                        "name": law,
                        "category": category,
                        "country": country["name"]
                    })
                    law_counter += 1

    # 2. Build the PROMPT_POOL dynamically
    sample_goods = {
        "Food": "Organic Cavendish Bananas",
        "Nuclear": "Uranium-235 Isotopes",
        "Pharmaceutical": "Amoxicillin Active Pharmaceutical Ingredients",
        "Electronics": "Lithium-ion Battery Packs"
    }
    
    for industry_block in raw_data:
        category = category_map.get(industry_block["industry"], "General")
        countries = industry_block.get("countries", [])
        
        if len(countries) >= 2:
            for _ in range(3): 
                origin, destination = random.sample(countries, 2)
                
                required_laws = [law for law in destination.get("laws", [])]
                red_herrings = [
                    law["name"] for law in available_laws 
                    if law["category"] != category
                ]
                selected_herrings = random.sample(red_herrings, min(2, len(red_herrings)))
                
                qty = f"{random.randint(50, 500)} units"
                item = sample_goods.get(category, "Industrial Cargo")
                
                # Dynamic regulator key lookup
                reg_key = f"{category.lower()}_regulator"
                
                prompt_pool.append({
                    "text": f"Shipping {qty} of {item} from {origin['name']} to {destination['name']}.",
                    "truth": {
                        "qty": qty,
                        "category": category,
                        "Destination": destination["name"],
                        "Origin": origin["name"],
                        
                        # Full Compliance Package Data
                        "dest_regulator": destination.get(reg_key, "N/A"),
                        "origin_regulator": origin.get(reg_key, "N/A"),
                        "import_rules": destination.get("import_rules", {}),
                        "export_rules": origin.get("export_rules", {}),
                        
                        # Ground Truth Laws
                        "required_laws": required_laws,
                        "red_herrings": selected_herrings
                    }
                })

    return available_laws, prompt_pool

AVAILABLE_LAWS, PROMPT_POOL = load_environment_data("data/final_dataset.json")


class CargoComplianceEnv(Environment):
    def __init__(self):
        self.sessions: Dict[str, Cargo_State] = {}
        self.last_session_id:  Optional[str] = None
    
    def reset(self, seed = None, options = None) -> Tuple[Cargo_Observation, Dict[str, Any]]:
        session_id, obs = self.create_task()
        self.last_session_id = session_id
        return obs, {"session_id":session_id}
    
    def state(self) -> Dict[str,Any]:
        if not self.last_session_id or self.last_session_id not in self.sessions:
            return {}
        
        state_obj = self.sessions[self.last_session_id]
        return {
            "phase":state_obj.phase,
            "steps":state_obj.steps,
            "questions_asked": state_obj.questions_asked,
            "extraction_data": state_obj.extraction_data,
            "total_reward": state_obj.total_reward,
        }
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_llm_judge_score(self, reasoning: str, extraction: dict, truth: dict) -> float:
        compliance_context = {
            "Expected_Regulator": truth["dest_regulator"],
            "Expected_Import_Docs": truth["import_rules"].get("documents", []),
            "Expected_Export_Docs": truth["export_rules"].get("documents", []),
            "Expected_Laws": truth["required_laws"]
        }
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict customs auditor. Grade the agent's reasoning from 0.0 to 1.0. "
                                   "Respond with ONLY a number. Example: 0.8"
                    },
                    {
                        "role": "user",
                        "content": f"Target Rules: {json.dumps(compliance_context)}\nAgent Reasoning: {reasoning}"
                    }
                ],
                # FIX: Use a valid Groq model ID
                model="mistral", 
                max_tokens=10,
                temperature=0,
            )
            
            raw_content = chat_completion.choices[0].message.content.strip()
            
            # FIX: Use regex to extract the float in case the LLM adds text
            score_match = re.search(r"(\d?\.\d+)", raw_content)
            if score_match:
                return float(score_match.group(1))
            return float(raw_content)

        except Exception as e:
            # DEBUG: See what's actually failing in your Fedora terminal
            print(f"JUDGE ERROR: {e}") 
            return 0.5

    def create_task(self) -> Tuple[str, Cargo_Observation]:
        session_id = str(uuid.uuid4())
        selected_task = random.choice(PROMPT_POOL)
        
        state = Cargo_State(
            task_id=session_id,
            steps=0,
            history=[],
            phase="EXTRACTION",
            questions_asked=0,
            total_reward=0.0,
            # Added regulator and documents to initial state
            extraction_data={"qty": None, "category": None, "Destination": None, "Origin": None, "laws": [], "regulator": None, "documents": []}
        )
        state.ground_truth = selected_task["truth"]
        self.sessions[session_id] = state
        
        initial_obs = Cargo_Observation(
            text=f"NEW SHIPMENT: {selected_task['text']}\nExtract into JSON: qty, category, Destination, Origin. Max 3 questions. Penalty: -0.1/question, -1.0/wrong guess.",
            current_extraction=state.extraction_data,
            available_laws=[],
            manifest={"raw_text": selected_task["text"]},
            laws=[],
            history=[],
            step=0,
            reward=0.0,
            total_reward=0.0
        )
        return session_id, initial_obs

    async def step(self, session_id: str, action: Cargo_Action) -> Cargo_Observation:
        state = self.sessions.get(session_id)
        if not state: 
            raise ValueError("Session not found.")

        truth = state.ground_truth
        step_reward = 0.0
        obs_text = ""
        state.steps += 1

        def clean(v): 
            return str(v or "").strip().lower()

        # --- PHASE 1: EXTRACTION ---
        if state.phase == "EXTRACTION":
            if action.action_type == Cargo_FetchState.FETCH_INFO:
                if state.questions_asked < 3:
                    state.questions_asked += 1
                    step_reward = -0.1
                    obs_text = f"Clarification provided. ({state.questions_asked}/3)"
                else:
                    obs_text = "Question limit reached. Submit JSON."
            
            elif action.action_type == Cargo_FetchState.VERDICT:
                try:
                    data = json.loads(action.decision)
                    if not isinstance(data, dict):
                        raise ValueError("Expected a JSON object for extraction.")
                    state.extraction_data.update(data)

                    ext_qty = clean(data.get("qty"))
                    truth_qty = clean(truth["qty"])
                    qty_match = ext_qty in truth_qty or truth_qty in ext_qty

                    is_correct = (
                        qty_match and 
                        clean(data.get("category")) == clean(truth["category"]) and
                        clean(data.get("Destination")) == clean(truth["Destination"]) and
                        clean(data.get("Origin")) == clean(truth["Origin"])
                    )
                    
                    if is_correct:
                        step_reward = 1.0 - (state.questions_asked * 0.1)
                        state.phase = "SELECTION"
                        obs_text = "Extraction Verified. Phase 2: Select Compliance Package."
                    else:
                        step_reward = -1.0
                        # Precise error logging for debugging
                        mismatches = []
                        if not qty_match: mismatches.append(f"Qty")
                        if clean(data.get("category")) != clean(truth["category"]): mismatches.append("Category")
                        if clean(data.get("Destination")) != clean(truth["Destination"]): mismatches.append("Destination")
                        if clean(data.get("Origin")) != clean(truth["Origin"]): mismatches.append("Origin")
                        obs_text = f"Extraction Failed: Mismatch in {', '.join(mismatches)}."
                        
                except (json.JSONDecodeError, TypeError, ValueError):
                    step_reward = -1.0
                    obs_text = "ERROR: Invalid JSON format. Please submit valid JSON."

        # --- PHASE 2: COMPLIANCE SELECTION ---
        elif state.phase == "SELECTION":
            if action.action_type == Cargo_FetchState.PICK_LAW:
                try:
                    decision = json.loads(action.decision)
                    if not isinstance(decision, dict):
                        raise ValueError("Expected a JSON object for compliance package.")
                    
                    # 1. Score Laws (Max +2.0 points)
                    selected_laws = decision.get("laws", [])
                    unique_selected_laws = list(dict.fromkeys(selected_laws))
                    state.extraction_data["laws"] = unique_selected_laws
                    
                    required_laws_set = set(truth["required_laws"])
                    selected_laws_set = set(unique_selected_laws)
                    
                    correct_laws = required_laws_set.intersection(selected_laws_set)
                    extra_laws = selected_laws_set - required_laws_set
                    
                    # Proportional reward: 100% correct gets +2.0
                    law_base_score = (len(correct_laws) / max(1, len(required_laws_set))) * 2.0
                    
                    # Penalties for hallucinations
                    law_penalty = 0.0
                    for law in extra_laws:
                        if law in truth["red_herrings"]:
                            law_penalty += 1.0  # Major penalty for falling for a trap
                        else:
                            law_penalty += 0.5  # Minor penalty for general hallucination
                            
                    # Bound the floor so one bad guess doesn't destroy the whole episode
                    step_reward += max(-1.0, law_base_score - law_penalty)

                    # 2. Score Regulator (+1.0 for correct, -0.5 for incorrect guess)
                    agent_regulator = decision.get("regulator", "")
                    state.extraction_data["regulator"] = agent_regulator
                    
                    if truth["dest_regulator"] != "N/A":
                        if clean(agent_regulator) == clean(truth["dest_regulator"]):
                            step_reward += 1.0
                        elif agent_regulator: # Penalize if they guessed wrong, but not if they left it blank
                            step_reward -= 0.5
                        
                    # 3. Score Documents (Max +2.0 points)
                    agent_docs = decision.get("documents", [])
                    unique_agent_docs = list(dict.fromkeys(agent_docs))
                    state.extraction_data["documents"] = unique_agent_docs
                    
                    all_required_docs = set(truth["import_rules"].get("documents", []) + 
                                            truth["export_rules"].get("documents", []))
                    
                    matched_required_docs = set()
                    for doc in unique_agent_docs:
                        # Stricter partial matching: require at least 5 chars to prevent "form" matching everything
                        clean_doc = clean(doc)
                        if len(clean_doc) > 4:
                            matched_doc = next(
                                (req_doc for req_doc in all_required_docs
                                 if clean_doc in clean(req_doc) or clean(req_doc) in clean_doc),
                                None,
                            )
                            if matched_doc:
                                matched_required_docs.add(matched_doc)

                    # Proportional reward: 100% correct gets +2.0
                    doc_base_score = (len(matched_required_docs) / max(1, len(all_required_docs))) * 2.0
                    
                    # Penalty only for the number of extra/irrelevant documents they submitted
                    doc_penalty = (len(unique_agent_docs) - len(matched_required_docs)) * 0.2
                    
                    step_reward += max(-1.0, doc_base_score - doc_penalty)
                    
                    state.phase = "VERDICT"
                    obs_text = "Compliance package verified. Final Step: Submit Reasoning."
                    
                except (json.JSONDecodeError, TypeError, ValueError):
                    step_reward = -1.0
                    obs_text = "ERROR: Provide laws, regulator, and documents in a JSON object."

        # --- PHASE 3: FINAL AUDIT ---
        elif state.phase == "VERDICT":
            if action.action_type == Cargo_FetchState.VERDICT:
                audit_score = await self.get_llm_judge_score(action.decision, state.extraction_data, truth)
                step_reward += audit_score * 2.0
                obs_text = f"Audit Complete. Judge Score: {audit_score}. Episode Finished."

        state.total_reward += step_reward

        # Filter laws for the LLM based on destination
        available_laws_subset = [
            law for law in AVAILABLE_LAWS 
            if clean(law["country"]) == clean(truth["Destination"])
        ] if state.phase == "SELECTION" else []

        return Cargo_Observation(
            text=obs_text,
            current_extraction=state.extraction_data,
            available_laws=available_laws_subset,
            manifest={},
            laws=state.extraction_data.get("laws", []),
            history=state.history,
            step=state.steps,
            reward=step_reward,
            total_reward=state.total_reward
        )
