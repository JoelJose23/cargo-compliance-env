from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Cargo_Action, Cargo_Observation, Cargo_State
from server.environment import CargoComplianceEnv # Import your real class

app = FastAPI(title="Cargo Compliance Production API")

# Enable CORS for the Hackathon dashboard/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your actual environment engine
env = CargoComplianceEnv()

@app.post("/reset")
async def reset():
    """Starts a real session using your dynamic PROMPT_POOL."""
    # env.reset() returns (observation, info_dict)
    obs, info = env.reset()
    return obs

@app.post("/step")
async def step(action: Cargo_Action, session_id: str = None):
    """Routes the UI/AI action to the real RL logic."""
    # Use the session_id from the header or query if your environment supports multi-user
    # For a single-user hackathon test, we use the last_session_id
    target_id = session_id or env.last_session_id
    
    if not target_id:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    
    try:
        # CRITICAL: This must be awaited because of the LLM Judge!
        obs = await env.step(target_id, action)
        
        # Determine if the episode is finished
        # Usually, Phase 3 (VERDICT) marks the end
        done = (obs.text and "Episode Finished" in obs.text)
        
        return {
            "observation": obs,
            "reward": obs.reward,
            "done": done,
            "total_reward": obs.total_reward
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def get_tasks():
    """Returns the JSON Schema so the AI knows how to format its actions."""
    schema = Cargo_Action.model_json_schema()
    return {
        "tasks": [
            {"id": "dynamic_cargo", "difficulty": "adaptive", "description": "Real-time compliance scenarios."}
        ],
        "action_schema": schema
    }

@app.get("/health")
async def health():
    return {"status": "online", "engine": "CargoComplianceEnv v1.0"}