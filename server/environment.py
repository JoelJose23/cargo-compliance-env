import random
import uuid
import json
import os
from typing import Dict, Any, Tuple, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from openenv.core.env_server import Environment
from .models import Cargo_Action, Cargo_Observation, Cargo_FetchState, Cargo_State
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Cargo Compliance Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_law_list(country: Dict[str, Any], rule_key: str) -> List[str]:
    """Return a non-empty law list for import/export scoring.

    Dataset variants may store laws under `country["laws"]` instead of
    `country["import_rules"]["laws"]` / `country["export_rules"]["laws"]`.
    """
    rule_laws = country.get(rule_key, {}).get("laws", [])
    if isinstance(rule_laws, list) and rule_laws:
        return rule_laws

    country_laws = country.get("laws", [])
    if isinstance(country_laws, list):
        return country_laws
    return []

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
    # Inside load_environment_data
    for industry_block in raw_data:
        category = category_map.get(industry_block["industry"], "General")
        for country in industry_block.get("countries", []):
            import_laws = _normalize_law_list(country, "import_rules")
            export_laws = _normalize_law_list(country, "export_rules")

            # Tag Import Laws
            for law in import_laws:
                available_laws.append({
                    "id": f"LAW_{law_counter:03d}",
                    "name": law,
                    "category": category,
                    "country": country["name"],
                    "type": "Import" # NEW TAG
                })
                law_counter += 1
            # Tag Export Laws
            for law in export_laws:
                available_laws.append({
                    "id": f"LAW_{law_counter:03d}",
                    "name": law,
                    "category": category,
                    "country": country["name"],
                    "type": "Export" # NEW TAG
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
            # Increased range to 5 to get a better mix of full and broken prompts
            for _ in range(5): 
                origin, destination = random.sample(countries, 2)
                
                required_export_laws = _normalize_law_list(origin, "export_rules")
                required_import_laws = _normalize_law_list(destination, "import_rules")
                all_required_laws = required_export_laws + required_import_laws
                
                red_herrings = [
                    law["name"] for law in available_laws 
                    if law["category"] != category
                ]
                selected_herrings = random.sample(red_herrings, min(2, len(red_herrings)))
                
                qty = f"{random.randint(50, 500)} units"
                item = sample_goods.get(category, "Industrial Cargo")
                reg_key = f"{category.lower()}_regulator"

                # --- NEW: PROMPT VARIANT LOGIC ---
                # This ensures the agent isn't always fed the answer on a silver platter.
                prompt_variants = [
                    f"Shipping {qty} of {item} from {origin['name']} to {destination['name']}.", # Perfect
                    f"Shipping {qty} of {item} to {destination['name']}.",                       # Missing Origin
                    f"I need to move {item} from {origin['name']} to {destination['name']}.",    # Missing Qty
                    f"New shipment of {qty} from {origin['name']} to {destination['name']}.",    # Missing Category
                    f"Requesting compliance check for cargo from {origin['name']} to {destination['name']}." # Barebones
                ]
                
                # Pick one variant randomly for this entry in the pool
                selected_text = random.choice(prompt_variants)
                
                prompt_pool.append({
                    "text": selected_text,
                    "truth": {
                        "qty": qty,
                        "category": category,
                        "Destination": destination["name"],
                        "Origin": origin["name"],
                        "dest_regulator": destination.get(reg_key, "N/A"),
                        "origin_regulator": origin.get(reg_key, "N/A"),
                        "import_rules": destination.get("import_rules", {}),
                        "export_rules": origin.get("export_rules", {}),
                        "required_export_laws": required_export_laws,
                        "required_import_laws": required_import_laws,
                        "all_required_laws": all_required_laws,
                        "red_herrings": selected_herrings
                    }
                })

    return available_laws, prompt_pool

_BASE_DIR = os.path.dirname(__file__)
_DATA_PATH = os.path.abspath(os.path.join(_BASE_DIR, "..", "data", "final_dataset.json"))
AVAILABLE_LAWS, PROMPT_POOL = load_environment_data(_DATA_PATH)


class CargoComplianceEnv(Environment):
    def __init__(self, base_url: str = "http://localhost:7860"):
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
            "Expected_Laws": truth.get("all_required_laws", [])
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
                model="llama-3.3-70b-versatile", 
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
            extraction_data={"qty": None, "category": None, "Destination": None, "Origin": None, "laws": [], "regulator": None, "documents": [], "duties": []}
        )
        state.ground_truth = selected_task["truth"]
        self.sessions[session_id] = state
        
        initial_obs = Cargo_Observation(
            text=f"NEW SHIPMENT: {selected_task['text']}\nExtract into JSON: qty, category, Destination, Origin. Max 3 questions. Penalty: -0.1/question, -1.0/wrong guess.",
            current_extraction=state.extraction_data,
            available_laws=[],
            manifest={"raw_text": selected_task["text"]},
            laws=[],
            documents=[],
            regulator=None,
            duties=[],
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

        # --- PHASE 1: EXTRACTION -
        if state.phase == "EXTRACTION":
            if action.action_type == Cargo_FetchState.FETCH_INFO:
                if state.questions_asked < 3:
                    state.questions_asked += 1
                    # The "Annoyance Cost": Asking a question costs a small amount of reward
                    step_reward = -0.1
                    
                    query = action.decision.lower()
                    responses = []
                    
                    # The Environment (Customer) checks if the Seller (Agent) is asking 
                    # about specific missing fields.
                    if "origin" in query:
                        responses.append(f"The shipment is originating from {truth['Origin']}")
                    if "qty" in query or "quantity" in query:
                        responses.append(f"The total quantity is {truth['qty']}")
                    if "category" in query:
                        responses.append(f"This falls under the {truth['category']} category")
                    if "destination" in query or "dest" in query:
                        responses.append(f"The final destination is {truth['Destination']}")

                    if responses:
                        # The "Customer" answers the question
                        customer_reply = " and ".join(responses)
                        obs_text = f"CUSTOMER REPLY: '{customer_reply}.' [Questions Used: {state.questions_asked}/3]"
                    else:
                        # The "Customer" is confused by the question
                        obs_text = f"CUSTOMER REPLY: 'I don't understand that question. I'm shipping cargo.' [Questions Used: {state.questions_asked}/3]"
                
                else:
                    # The Customer is fed up and refuses to answer more
                    step_reward = -0.2
                    obs_text = "CUSTOMER: 'I've answered enough questions. Please just process the shipment!'"
        
            elif action.action_type == Cargo_FetchState.SUBMIT_EXTRACT:
                try:
                    data = json.loads(action.decision)
                    if not isinstance(data, dict):
                        raise ValueError("Expected a JSON object for extraction.")
                    state.extraction_data.update(data)

                    # --- DENSE REWARD LOGIC ---
                    fields = ["qty", "category", "Destination", "Origin"]
                    correct_count = 0
                    mismatches = []
                    
                    # 1. Check Qty (Fuzzy match)
                    ext_qty = clean(data.get("qty"))
                    truth_qty = clean(truth["qty"])
                    if ext_qty in truth_qty or truth_qty in ext_qty:
                        correct_count += 1
                    else:
                        mismatches.append("Qty")
                        
                    # 2. Check Standard Fields
                    for f in ["category", "Destination", "Origin"]:
                        if clean(data.get(f)) == clean(truth[f]):
                            correct_count += 1
                        else:
                            mismatches.append(f)

                    # Calculate Partial Credit (0.25 per correct field)
                    base_reward = (correct_count / len(fields)) * 1.0
                    
                    # Apply "Cost of Living" penalty for questions asked
                    step_reward = base_reward - (state.questions_asked * 0.1)

                    if correct_count == len(fields):
                        state.phase = "SELECTION"
                        obs_text = "Extraction Verified. Phase 2: Select Compliance Package."
                    else:
                        # Don't fail immediately, let them try again, but penalize the mismatch
                        step_reward -= 0.5 
                        obs_text = f"Extraction Partial/Failed: Mismatch in {', '.join(mismatches)}."
                        
                except (json.JSONDecodeError, TypeError, ValueError):
                    step_reward = -1.0
                    obs_text = "ERROR: Invalid JSON format. Please submit valid JSON."

        # --- PHASE 2: COMPLIANCE SELECTION (Bilateral Update) ---
        elif state.phase == "SELECTION":
            if action.action_type == Cargo_FetchState.PICK_LAW:
                try:
                    decision = json.loads(action.decision)
                    if not isinstance(decision, dict):
                        raise ValueError("Expected a JSON object.")
                    
                    # 1. Score Laws (Bilateral: Export + Import)
                    selected_laws = decision.get("laws", [])
                    unique_selected_laws = list(dict.fromkeys(selected_laws))
                    state.extraction_data["laws"] = unique_selected_laws
                    
                    selected_laws_set = set(unique_selected_laws)
                    
                    # Split Ground Truth sets
                    export_truth = set(truth.get("required_export_laws", []))
                    import_truth = set(truth.get("required_import_laws", []))
                    red_herrings = set(truth.get("red_herrings", []))

                    # Calculate Matches
                    export_matches = selected_laws_set.intersection(export_truth)
                    import_matches = selected_laws_set.intersection(import_truth)
                    
                    # Scoring: 1.0 for Export, 1.0 for Import (Total +2.0)
                    export_score = (len(export_matches) / max(1, len(export_truth))) * 1.0
                    import_score = (len(import_matches) / max(1, len(import_truth))) * 1.0
                    
                    # Hallucination Penalty
                    extra_laws = selected_laws_set - export_truth - import_truth
                    law_penalty = sum(0.3 if law in red_herrings else 0.5 for law in extra_laws)
                    
                    step_reward += max(0.0, (export_score + import_score) - law_penalty)

                    # 2. Score Regulators (+1.0 Total: 0.5 for Origin, 0.5 for Dest)
                    agent_regulator = clean(decision.get("regulator", ""))
                    state.extraction_data["regulator"] = agent_regulator
                    
                    reg_score = 0.0
                    # Check Origin Regulator
                    if truth["origin_regulator"] != "N/A" and truth["origin_regulator"].lower() in agent_regulator:
                        reg_score += 0.5
                    # Check Destination Regulator
                    if truth["dest_regulator"] != "N/A" and truth["dest_regulator"].lower() in agent_regulator:
                        reg_score += 0.5
                    
                    step_reward += reg_score
                        
                    # 3. Score Documents (Max +2.0 points)
                    agent_docs = decision.get("documents", [])
                    unique_agent_docs = list(dict.fromkeys(agent_docs))
                    state.extraction_data["documents"] = unique_agent_docs
                    
                    # Combine Export & Import Docs for the truth set
                    all_required_docs = set(truth["import_rules"].get("documents", []) + 
                                            truth["export_rules"].get("documents", []))
                    
                    matched_required_docs = set()
                    for doc in unique_agent_docs:
                        clean_doc = clean(doc)
                        if len(clean_doc) > 4:
                            matched_doc = next(
                                (req_doc for req_doc in all_required_docs
                                 if clean_doc in clean(req_doc) or clean(req_doc) in clean_doc),
                                None,
                            )
                            if matched_doc:
                                matched_required_docs.add(matched_doc)

                    doc_base_score = (len(matched_required_docs) / max(1, len(all_required_docs))) * 2.0
                    doc_penalty = min(0.5, (len(unique_agent_docs) - len(matched_required_docs)) * 0.1)
                    
                    step_reward += max(0.0, doc_base_score - doc_penalty)
                    
                    state.phase = "VERDICT"
                    obs_text = "Bilateral compliance verified. Final Step: Submit Reasoning."
                    
                except (json.JSONDecodeError, TypeError, ValueError):
                    step_reward = -1.0
                    obs_text = "ERROR: Provide laws, regulator, and documents in a JSON object."
                    
        # --- PHASE 3: FINAL AUDIT ---
        elif state.phase == "VERDICT":
            if action.action_type == Cargo_FetchState.FINAL_VERDICT:
                audit_score = await self.get_llm_judge_score(action.decision, state.extraction_data, truth)
                step_reward += audit_score * 2.0
                obs_text = f"Audit Complete. Judge Score: {audit_score}. Episode Finished."

        state.total_reward += step_reward

        # Filter laws for the LLM based on destination
        # Filter laws for BOTH countries
        available_laws_subset = [
            law for law in AVAILABLE_LAWS 
            if (clean(law["country"]) == clean(truth["Origin"]) and law["type"] == "Export") or 
            (clean(law["country"]) == clean(truth["Destination"]) and law["type"] == "Import")
        ] if state.phase == "SELECTION" else []

        return Cargo_Observation(
            text=obs_text,
            current_extraction=state.extraction_data,
            available_laws=available_laws_subset,
            manifest={},
            documents= state.extraction_data.get("documents", []),
            regulator=state.extraction_data.get("regulator"),
            duties=state.extraction_data.get("duties", []),
            laws=state.extraction_data.get("laws", []),
            history=state.history,
            step=state.steps,
            reward=step_reward,
            total_reward=state.total_reward
        )


# --- API wiring ---
env = CargoComplianceEnv()


@app.post("/reset")
async def reset():
    """Initialize a session and return the starting observation."""
    obs, info = env.reset()
    return obs


@app.post("/step")
async def step(action: Cargo_Action, session_id: str = None) -> Dict[str, Any]:
    target_id = session_id or env.last_session_id

    if not target_id:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    try:
        obs = await env.step(target_id, action)
        done = bool(obs.text and "Episode Finished" in obs.text)
        return {
            "observation": obs,
            "reward": obs.reward,
            "done": done,
            "total_reward": obs.total_reward,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {"id": "cargo_food", "description": "Food Industry Compliance"},
            {"id": "cargo_electronics", "description": "Electronics Export Control"},
            {"id": "cargo_pharma", "description": "Pharmaceutical Regulations"}
        ],
        "action_schema": Cargo_Action.model_json_schema(),
    }

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "online", "engine": "CargoComplianceEnv v1.0"}
