import random
import uuid
import json
import os
from typing import Dict, Any, Tuple, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openenv.core.env_server import Environment

try:
    from models import Cargo_Action, Cargo_Observation, Cargo_FetchState, Cargo_State
except ImportError:
    from models import Cargo_Action, Cargo_Observation, Cargo_FetchState, Cargo_State
import re
from dotenv import load_dotenv

load_dotenv()

# Define the expected JSON payload for session initialization
class ResetRequest(BaseModel):
    task_id: Optional[str] = None

app = FastAPI(title="Cargo Compliance Production API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_law_list(country: Dict[str, Any], rule_key: str) -> List[str]:
    """
    Utility: Return a non-empty law list for import/export scoring.
    Handles variations in dataset structures where laws might be nested under
    specific rule sets or directly at the country level.
    """
    rule_laws = country.get(rule_key, {}).get("laws", [])
    if isinstance(rule_laws, list) and rule_laws:
        return rule_laws

    country_laws = country.get("laws", [])
    if isinstance(country_laws, list):
        return country_laws
    return []


# --- CORE ENGINE: Data Loading & Prompt Generation ---
def load_environment_data(json_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Loads the static JSON dataset and dynamically generates prompt variants.
    This ensures agents cannot memorize static text and must actively use 
    tools (like FETCH_INFO) to retrieve missing context.
    """
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    available_laws = []
    prompt_pool = []

    # Map dataset industries to our standard task categories
    category_map = {
        "Food Products (Processed, Packaged, Agricultural)": "Food",
        "Radioactive Materials and Nuclear Goods": "Nuclear",
        "Pharmaceutical Drugs (Finished Formulations, APIs, Controlled Substances)": "Pharmaceutical",
        "Electronics (Consumer, Industrial, and Dual-use Goods)": "Electronics",
    }

    # 1. Build the AVAILABLE_LAWS registry (Global Knowledge Base)
    law_counter = 1
    for industry_block in raw_data:
        category = category_map.get(industry_block["industry"], "General")
        for country in industry_block.get("countries", []):
            import_laws = _normalize_law_list(country, "import_rules")
            export_laws = _normalize_law_list(country, "export_rules")

            # Tag and register Import Laws
            for law in import_laws:
                available_laws.append(
                    {
                        "id": f"LAW_{law_counter:03d}",
                        "name": law,
                        "category": category,
                        "country": country["name"],
                        "type": "Import", 
                    }
                )
                law_counter += 1
                
            # Tag and register Export Laws
            for law in export_laws:
                available_laws.append(
                    {
                        "id": f"LAW_{law_counter:03d}",
                        "name": law,
                        "category": category,
                        "country": country["name"],
                        "type": "Export", 
                    }
                )
                law_counter += 1

    # 2. Build the PROMPT_POOL dynamically
    sample_goods = {
        "Food": "Organic Cavendish Bananas",
        "Nuclear": "Uranium-235 Isotopes",
        "Pharmaceutical": "Amoxicillin Active Pharmaceutical Ingredients",
        "Electronics": "Lithium-ion Battery Packs",
    }

    for industry_block in raw_data:
        category = category_map.get(industry_block["industry"], "General")
        countries = industry_block.get("countries", [])

        if len(countries) >= 2:
            # Generate 5 variants per country pair to test agent robustness
            for _ in range(5):
                origin, destination = random.sample(countries, 2)

                required_export_laws = _normalize_law_list(origin, "export_rules")
                required_import_laws = _normalize_law_list(destination, "import_rules")
                all_required_laws = required_export_laws + required_import_laws

                # Inject red herrings (laws from wrong categories) to test hallucination resistance
                red_herrings = [
                    law["name"] for law in available_laws if law["category"] != category
                ]
                selected_herrings = random.sample(
                    red_herrings, min(2, len(red_herrings))
                )

                qty = f"{random.randint(50, 500)} units"
                item = sample_goods.get(category, "Industrial Cargo")
                reg_key = f"{category.lower()}_regulator"

                # Prompt Variant Logic: Intentionally obscure data to force tool usage
                prompt_variants = [
                    f"Shipping {qty} of {item} from {origin['name']} to {destination['name']}.",  # Perfect
                    f"Shipping {qty} of {item} to {destination['name']}.",  # Missing Origin
                    f"I need to move {item} from {origin['name']} to {destination['name']}.",  # Missing Qty
                    f"New shipment of {qty} from {origin['name']} to {destination['name']}.",  # Missing Category
                    f"Requesting compliance check for cargo from {origin['name']} to {destination['name']}.",  # Barebones
                ]

                selected_text = random.choice(prompt_variants)

                prompt_pool.append(
                    {
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
                            "red_herrings": selected_herrings,
                        },
                    }
                )

    return available_laws, prompt_pool

# Initialize global dataset state
_BASE_DIR = os.path.dirname(__file__)
_DATA_PATH = os.path.abspath(
    os.path.join(_BASE_DIR, "..", "data", "final_dataset.json")
)
AVAILABLE_LAWS, PROMPT_POOL = load_environment_data(_DATA_PATH)

# Define task specifications with strict pass thresholds
TASK_SPECS = {
    "cargo_food": {
        "category": "Food",
        "difficulty": "easy",
        "description": "Complete bilateral food-compliance screening for an agricultural shipment.",
        "objective": "Extract shipment basics, ask for missing details only when needed, and select the exact food import/export compliance package.",
        "pass_score": 0.70,
    },
    "cargo_electronics": {
        "category": "Electronics",
        "difficulty": "medium",
        "description": "Resolve export-control obligations for a dual-use electronics shipment.",
        "objective": "Identify the correct origin/destination details and choose the matching electronics laws, regulators, and documents.",
        "pass_score": 0.78,
    },
    "cargo_pharma": {
        "category": "Pharmaceutical",
        "difficulty": "hard",
        "description": "Validate pharmaceutical API compliance across origin and destination jurisdictions.",
        "objective": "Handle stricter pharma extraction and select the exact controlled-substance paperwork without hallucinating extra laws.",
        "pass_score": 0.85,
    },
}

CATEGORY_TO_TASK = {spec["category"]: task_id for task_id, spec in TASK_SPECS.items()}
SUPPORTED_CATEGORIES = set(CATEGORY_TO_TASK.keys())


class CargoComplianceEnv(Environment):
    """
    Stateful RL Environment for Cargo Compliance.
    Manages user sessions, state transitions, and deterministic grading.
    """
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.sessions: Dict[str, Cargo_State] = {}
        self.last_session_id: Optional[str] = None

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Cargo_Observation, Dict[str, Any]]:
        task_id = options.get("task_id") if options else None
        session_id, obs = self.create_task(task_id=task_id)
        self.last_session_id = session_id
        return obs, {"session_id": session_id}

    def state(self) -> Dict[str, Any]:
        """Returns the current internal state metrics for the active session."""
        if not self.last_session_id or self.last_session_id not in self.sessions:
            return {}

        state_obj = self.sessions[self.last_session_id]
        return {
            "phase": state_obj.phase,
            "steps": state_obj.steps,
            "questions_asked": state_obj.questions_asked,
            "extraction_data": state_obj.extraction_data,
            "total_reward": state_obj.total_reward,
        }

    async def get_programmatic_grade(self, extraction: dict, truth: dict) -> float:
        """
        Deterministic Grader (0.01 - 0.99 range).
        Evaluates extraction accuracy, law selection (with hallucination penalties),
        regulator identification, and document completeness.
        """
        extraction = extraction or {}

        def clean(v: Any) -> str:
            return str(v or "").strip().lower()

        # Score 1: Extraction Match (Fuzzy for qty, exact for others)
        qty_truth = clean(truth.get("qty"))
        qty_guess = clean(extraction.get("qty"))
        qty_score = (
            1.0
            if qty_guess and (qty_guess in qty_truth or qty_truth in qty_guess)
            else 0.0
        )

        extraction_fields = (
            qty_score,
            1.0 if clean(extraction.get("category")) == clean(truth.get("category")) else 0.0,
            1.0 if clean(extraction.get("Destination")) == clean(truth.get("Destination")) else 0.0,
            1.0 if clean(extraction.get("Origin")) == clean(truth.get("Origin")) else 0.0,
        )
        extraction_score = sum(extraction_fields) / 4.0

        # Score 2: Law Selection (Penalize red herrings heavily)
        selected_laws = set(extraction.get("laws", []))
        required_laws = set(truth.get("all_required_laws", []))
        if not required_laws:
            required_laws = set(truth.get("required_export_laws", [])) | set(
                truth.get("required_import_laws", [])
            )
        
        law_match_score = len(selected_laws.intersection(required_laws)) / max(1, len(required_laws))
        law_extras = selected_laws - required_laws
        red_herrings = set(truth.get("red_herrings", []))
        # Strict for Safety: Red Herrings are a total failure
        red_herring_penalty = sum(1.0 for law in law_extras if law in red_herrings)
        # Moderate for Efficiency: Extra (but real) laws are just "over-compliance"
        over_compliance_penalty = sum(0.2 for law in law_extras if law not in red_herrings)

        law_score = max(0.0, law_match_score - (red_herring_penalty + over_compliance_penalty))

        # Score 3: Regulators
        regulator_targets = [truth.get("origin_regulator"), truth.get("dest_regulator")]
        regulator_targets = [clean(reg) for reg in regulator_targets if reg and reg != "N/A"]
        regulator_guess = clean(extraction.get("regulator"))
        regulator_hits = sum(1 for reg in regulator_targets if reg and reg in regulator_guess)
        regulator_score = regulator_hits / max(1, len(regulator_targets))

        # Score 4: Required Documents
        selected_docs = extraction.get("documents", []) or []
        required_docs = truth.get("import_rules", {}).get("documents", []) + truth.get("export_rules", {}).get("documents", [])
        matched_docs = set()
        
        for doc in selected_docs:
            clean_doc = clean(doc)
            if not clean_doc: continue
            for req_doc in required_docs:
                req_clean = clean(req_doc)
                if clean_doc in req_clean or req_clean in clean_doc:
                    matched_docs.add(req_doc)
        document_score = len(matched_docs) / max(1, len(required_docs))

        # Weighted Final Calculation
        final_score = (
            0.25 * extraction_score
            + 0.35 * law_score
            + 0.20 * regulator_score
            + 0.20 * document_score
        )
        return round(max(0.05, min(0.99, final_score)), 2)

    def create_task(self, task_id: Optional[str] = None) -> Tuple[str, Cargo_Observation]:
        """Initializes a new task session and constructs the ground truth state."""
        session_id = str(uuid.uuid4())
        final_task_id = None
        selected_task = None

        if task_id in TASK_SPECS:
            final_task_id = task_id
            target_category = TASK_SPECS[final_task_id]["category"]
            filtered_pool = [p for p in PROMPT_POOL if p["truth"]["category"] == target_category]
            selected_task = random.choice(filtered_pool) if filtered_pool else random.choice(PROMPT_POOL)
        else:
            supported_pool = [p for p in PROMPT_POOL if p["truth"]["category"] in SUPPORTED_CATEGORIES]
            selected_task = random.choice(supported_pool) if supported_pool else random.choice(PROMPT_POOL)
            final_task_id = CATEGORY_TO_TASK.get(selected_task["truth"]["category"], "cargo_food")

        state = Cargo_State(
            task_id=session_id,
            steps=0,
            history=[],
            phase="EXTRACTION", # Initial Phase
            questions_asked=0,
            total_reward=0.0,
            extraction_data={
                "qty": None, "category": None, "Destination": None, "Origin": None,
                "laws": [], "regulator": None, "documents": [], "duties": [],
            },
        )

        state.task_id_name = final_task_id
        state.ground_truth = selected_task["truth"]
        self.sessions[session_id] = state

        initial_obs = Cargo_Observation(
            text=f"NEW SHIPMENT: {selected_task['text']}\nExtract into JSON: qty, category, Destination, Origin. Max 3 questions. Penalty: -0.1/question, -1.0/wrong guess.",
            current_extraction=state.extraction_data,
            available_laws=[], available_documents=[], available_regulators=[],
            manifest={"raw_text": selected_task["text"]},
            laws=[], documents=[], regulator=None, duties=[], history=[],
            step=0, reward=0.0, total_reward=0.0,
        )
        return session_id, initial_obs

    async def step(self, session_id: str, action: Cargo_Action) -> Cargo_Observation:
        """
        The core state machine. Handles agent actions, updates state, and provides 
        dense rewards based on the current phase (EXTRACTION -> SELECTION -> VERDICT).
        """
        state = self.sessions.get(session_id)
        if not state: raise ValueError("Session not found.")

        truth = state.ground_truth
        step_reward = 0.0
        obs_text = ""
        grader_score = None
        state.steps += 1

        def clean(v): return str(v or "").strip().lower()

        # =====================================================================
        # PHASE 1: EXTRACTION (Data collection and tool usage)
        # =====================================================================
        if state.phase == "EXTRACTION":
            
            # Action: Agent asks a clarifying question
            if action.action_type == Cargo_FetchState.FETCH_INFO:
                if state.questions_asked < 3:
                    state.questions_asked += 1
                    step_reward = -0.1 # Annoyance Cost penalty

                    query = action.decision.lower()
                    responses = []

                    # Dynamic response generator based on agent inquiry
                    if "origin" in query: responses.append(f"The shipment is originating from {truth['Origin']}")
                    if "qty" in query or "quantity" in query: responses.append(f"The total quantity is {truth['qty']}")
                    if "category" in query: responses.append(f"This falls under the {truth['category']} category")
                    if "destination" in query or "dest" in query: responses.append(f"The final destination is {truth['Destination']}")

                    if responses:
                        customer_reply = " and ".join(responses)
                        obs_text = f"CUSTOMER REPLY: '{customer_reply}.' [Questions Used: {state.questions_asked}/3]"
                    else:
                        obs_text = f"CUSTOMER REPLY: 'I don't understand that question. I'm shipping cargo.' [Questions Used: {state.questions_asked}/3]"

                else:
                    step_reward = -0.2
                    obs_text = "CUSTOMER: 'I've answered enough questions. Please just process the shipment!'"

            # Action: Agent submits extraction for verification
            elif action.action_type == Cargo_FetchState.SUBMIT_EXTRACT:
                try:
                    data = json.loads(action.decision)
                    if not isinstance(data, dict): raise ValueError("Expected a JSON object for extraction.")
                    state.extraction_data.update(data)

                    # --- Dense Reward Calculation ---
                    fields = ["qty", "category", "Destination", "Origin"]
                    correct_count = 0
                    mismatches = []

                    # 1. Fuzzy match for Quantity
                    ext_qty = clean(data.get("qty"))
                    truth_qty = clean(truth["qty"])
                    if ext_qty and (ext_qty in truth_qty or truth_qty in ext_qty): correct_count += 1
                    else: mismatches.append("Qty")

                    # 2. Exact match for metadata
                    for f in ["category", "Destination", "Origin"]:
                        ext_val = clean(data.get(f))
                        if ext_val and ext_val == clean(truth[f]): correct_count += 1
                        else: mismatches.append(f)

                    # Partial Credit logic minus tool usage costs
                    base_reward = (correct_count / len(fields)) * 1.0
                    step_reward = base_reward - (state.questions_asked * 0.1)

                    # Phase Transition Gate
                    if correct_count == len(fields):
                        state.phase = "SELECTION"
                        obs_text = "Extraction Verified. Phase 2: Select Compliance Package."
                    else:
                        step_reward -= 0.5
                        obs_text = f"Extraction Failed: Missing or incorrect data . SYSTEM DIRECTIVE: Do not guess. You MUST use the 'FETCH_INFO' tool to ask the customer for this missing data."

                except (json.JSONDecodeError, TypeError, ValueError):
                    step_reward = -1.0
                    obs_text = "ERROR: Invalid JSON format. Please submit valid JSON."

        # =====================================================================
        # PHASE 2: COMPLIANCE SELECTION (Bilateral checks)
        # =====================================================================
        elif state.phase == "SELECTION":
            if action.action_type == Cargo_FetchState.PICK_LAW:
                try:
                    decision = json.loads(action.decision)
                    if not isinstance(decision, dict): raise ValueError("Expected a JSON object.")

                    # 1. Score Bilateral Laws
                    selected_laws = decision.get("laws", [])
                    unique_selected_laws = list(dict.fromkeys(selected_laws))
                    state.extraction_data["laws"] = unique_selected_laws
                    selected_laws_set = set(unique_selected_laws)

                    export_truth = set(truth.get("required_export_laws", []))
                    import_truth = set(truth.get("required_import_laws", []))
                    red_herrings = set(truth.get("red_herrings", []))

                    export_score = (len(selected_laws_set.intersection(export_truth)) / max(1, len(export_truth))) * 1.0
                    import_score = (len(selected_laws_set.intersection(import_truth)) / max(1, len(import_truth))) * 1.0

                    extra_laws = selected_laws_set - export_truth - import_truth
                    law_penalty = sum(0.5 if law in red_herrings else 0.1 for law in extra_laws)

                    step_reward += max(0.0, (export_score + import_score) - law_penalty)

                    # 2. Score Regulators
                    agent_regulator = clean(decision.get("regulator", ""))
                    state.extraction_data["regulator"] = agent_regulator
                    reg_score = 0.0
                    
                    if truth["origin_regulator"] != "N/A" and truth["origin_regulator"].lower() in agent_regulator: reg_score += 0.5
                    if truth["dest_regulator"] != "N/A" and truth["dest_regulator"].lower() in agent_regulator: reg_score += 0.5
                    step_reward += reg_score

                    # 3. Score Documents
                    agent_docs = decision.get("documents", [])
                    unique_agent_docs = list(dict.fromkeys(agent_docs))
                    state.extraction_data["documents"] = unique_agent_docs

                    all_required_docs = set(truth["import_rules"].get("documents", []) + truth["export_rules"].get("documents", []))
                    matched_required_docs = set()
                    
                    for doc in unique_agent_docs:
                        clean_doc = clean(doc)
                        if len(clean_doc) > 4:
                            matched_doc = next((req_doc for req_doc in all_required_docs if clean_doc in clean(req_doc) or clean(req_doc) in clean_doc), None)
                            if matched_doc: matched_required_docs.add(matched_doc)

                    doc_base_score = (len(matched_required_docs) / max(1, len(all_required_docs))) * 2.0
                    doc_penalty = min(0.5, (len(unique_agent_docs) - len(matched_required_docs)) * 0.1)

                    step_reward += max(0.0, doc_base_score - doc_penalty)

                    # Phase Transition Gate
                    state.phase = "VERDICT"
                    obs_text = "Bilateral compliance verified. Final Step: Submit Reasoning."

                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    step_reward = -0.5
                    obs_text = f"SYSTEM ERROR: Invalid submission format. Expected JSON keys: 'laws', 'regulator', 'documents'. Detail: {str(e)}"

        # =====================================================================
        # PHASE 3: FINAL AUDIT (Grading)
        # =====================================================================
        elif state.phase == "VERDICT":
            if action.action_type == Cargo_FetchState.FINAL_VERDICT:
                grader_score = await self.get_programmatic_grade(state.extraction_data, truth)
                step_reward += grader_score
                obs_text = f"Audit Complete. Programmatic Grade: {grader_score}. Episode Finished."

        # --- Difficulty Multiplier Logic ---
        difficulty_multiplier = 1.0
        current_task = getattr(state, "task_id_name", "cargo_food")

        # Scale penalties based on the task track
        if "pharma" in current_task:  
            if step_reward < 0: step_reward *= 1.5  # Hard mode: 50% stricter penalties
            if action.action_type == Cargo_FetchState.SUBMIT_EXTRACT and correct_count < len(fields):
                step_reward = -0.5  # Hard mode: No partial credit on bad extractions
        elif "electronics" in current_task:  
            if step_reward < 0: step_reward *= 1.2  # Medium mode: 20% stricter penalties
        
        actual_step_reward = step_reward * difficulty_multiplier
        state.total_reward += actual_step_reward

        # --- Dynamic Context Delivery ---
        # Only inject Laws and Documents into the observation once Phase 1 is cleared
        available_laws_subset = (
            [
                law for law in AVAILABLE_LAWS
                if (clean(law["country"]) == clean(truth["Origin"]) and law["type"] == "Export")
                or (clean(law["country"]) == clean(truth["Destination"]) and law["type"] == "Import")
            ]
            if state.phase == "SELECTION" else []
        )
        
        available_documents = (
            list(dict.fromkeys(truth.get("import_rules", {}).get("documents", []) + truth.get("export_rules", {}).get("documents", [])))
            if state.phase == "SELECTION" else []
        )
        
        available_regulators = (
            [regulator for regulator in [truth.get("origin_regulator"), truth.get("dest_regulator")] if regulator and regulator != "N/A"]
            if state.phase == "SELECTION" else []
        )

        return Cargo_Observation(
            text=obs_text,
            current_extraction=state.extraction_data,
            available_laws=available_laws_subset,
            available_documents=available_documents,
            available_regulators=available_regulators,
            manifest={},
            documents=state.extraction_data.get("documents", []),
            regulator=state.extraction_data.get("regulator"),
            duties=state.extraction_data.get("duties", []),
            laws=state.extraction_data.get("laws", []),
            history=state.history,
            step=state.steps,
            reward=actual_step_reward,
            total_reward=state.total_reward,
            grader_score=grader_score,
        )


# --- API Routing ---
env = CargoComplianceEnv()

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """API Endpoint: Initialize a new scenario."""
    req_task_id = request.task_id if request else None
    obs, info = env.reset(options={"task_id": req_task_id})
    return {"observation": obs, "session_id": info["session_id"]}

@app.post("/step")
async def step(action: Cargo_Action, session_id: str = None) -> Dict[str, Any]:
    """API Endpoint: Process agent action and step the environment state."""
    target_id = session_id or env.last_session_id
    if not target_id:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    try:
        obs = await env.step(target_id, action)
        done = bool(obs.text and "Episode Finished" in obs.text)
        return {"observation": obs, "reward": obs.reward, "done": done, "total_reward": obs.total_reward}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/tasks")
async def get_tasks() -> List[Dict[str, Any]]:
    """API Endpoint: Expose benchmark requirements to the runner."""
    return [
        {
            "id": task_id, "task_id": task_id, "name": task_id,
            "description": spec["description"], "objective": spec["objective"],
            "difficulty": spec["difficulty"], "grader": "deterministic_programmatic",
            "grader_type": "programmatic", "has_grader": True,
            "score_range": [0.01, 0.99], "pass_score": spec["pass_score"],
        }
        for task_id, spec in TASK_SPECS.items()
    ]

@app.get("/metadata")
async def metadata() -> Dict[str, str]:
    return {"name": "cargo-compliance-challenge", "description": "Deterministic bilateral cargo-compliance benchmark with three graded tasks."}

@app.get("/schema")
async def schema() -> Dict[str, Any]:
    return {"action": Cargo_Action.model_json_schema(), "observation": Cargo_Observation.model_json_schema(), "state": Cargo_State.model_json_schema()}

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "engine": "CargoComplianceEnv v1.0"}
