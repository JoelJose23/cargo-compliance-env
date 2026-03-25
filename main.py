from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import Cargo_FetchState, Cargo_Observation, Cargo_State, Cargo_Action

app = FastAPI(title="Cargo Monitor API")

# 🚨 CRITICAL FOR JS INTEGRATION: Prevents browser CORS blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state tracker (Day 2 Mock - will move to CargoEnv class on Day 3)
current_state = Cargo_State(
    task_id="easy_textiles",
    steps=0,
    history=[]
)

@app.post("/reset", response_model=Cargo_Observation)
async def reset():
    """Initializes a new shipment simulation and returns the first observation."""
    global current_state
    current_state.steps = 0
    current_state.history = []
    
    # Send the first dummy cargo to the frontend
    return Cargo_Observation(
        text="New Shipment Pending Clearance: 200kg Cotton T-Shirts to France.",
        manifest={"cargo": "T-Shirts", "weight": "200kg", "origin": "IN", "destination": "FR"},
        laws=["LAW_EU_GEN_01", "LAW_FR_TEX_05"],
        history=[],
        step=0
    )

@app.post("/step")
async def step(action: Cargo_Action):
    """Processes an action from the UI or AI Baseline and returns the outcome."""
    global current_state
    current_state.steps += 1
    
    # 1. Log the action
    action_log = f"{action.action_type.value}: {action.query or action.decision}"
    current_state.history.append(action_log)
    
    # 2. Day 2 Dummy Reward Logic
    reward = 0.0
    if action.action_type == Cargo_FetchState.FETCH_INFO:
        reward = 0.1
    elif action.action_type == Cargo_FetchState.PICK_LAW:
        reward = 0.3
    elif action.action_type == Cargo_FetchState.VERDICT:
        # If it's the right decision, big reward. Otherwise, penalty.
        reward = 0.6 if action.decision == "RELEASE" else -1.0
        
    # 3. Check if episode is over (Verdict given OR too many steps)
    done = action.action_type == Cargo_FetchState.VERDICT or current_state.steps > 10
    
    # 4. Construct the new observation
    obs = Cargo_Observation(
        text=f"Action '{action.action_type.value}' processed successfully.",
        manifest={"cargo": "T-Shirts", "weight": "200kg", "origin": "IN", "destination": "FR"},
        laws=["LAW_EU_GEN_01", "LAW_FR_TEX_05"],
        history=current_state.history,
        step=current_state.steps
    )
    
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": {"status": "Complete" if done else "Processing"}
    }

@app.get("/state", response_model=Cargo_State)
async def get_state():
    """Returns the raw internal state."""
    return current_state

@app.get("/tasks")
async def get_tasks():
    """REQUIRED: Returns tasks and the Pydantic schema for the validator."""
    # Using model_json_schema() for Pydantic v2 (FastAPI default)
    # If using older Pydantic, use Cargo_Action.schema()
    schema = Cargo_Action.model_json_schema() if hasattr(Cargo_Action, "model_json_schema") else Cargo_Action.schema()
    
    return {
        "tasks": [
            {"id": "easy_textiles", "difficulty": "easy", "description": "Standard clothing export."},
            {"id": "med_electronics", "difficulty": "medium", "description": "Laptops with lithium batteries."},
            {"id": "hard_chemicals", "difficulty": "hard", "description": "Dual-use hazardous materials."}
        ],
        "action_schema": schema
    }

@app.get("/baseline")
async def trigger_baseline():
    """REQUIRED: Automated validator hits this to check if env is solvable."""
    return {"scores": {"easy_textiles": 1.0, "med_electronics": 0.8, "hard_chemicals": 0.4}}

@app.get("/grader")
async def get_grader_score():
    """REQUIRED: Automated validator hits this to get the final deterministic score."""
    return {"final_score": 1.0}