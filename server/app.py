import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import Cargo_Action, Cargo_Observation, Cargo_State
from .environment import CargoComplianceEnv

app = FastAPI(title="Cargo Compliance Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = CargoComplianceEnv()

@app.post("/reset")
async def reset():
    obs, info = env.reset()
    return obs

@app.post("/step")
async def step(action: Cargo_Action, session_id: str = None):
    target_id = session_id or env.last_session_id
    if not target_id:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    
    try:
        obs = await env.step(target_id, action)
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
    schema = Cargo_Action.model_json_schema()
    return {
        "tasks": [{"id": "dynamic_cargo", "difficulty": "adaptive", "description": "Real-time compliance scenarios."}],
        "action_schema": schema
    }

@app.get("/health")
async def health():
    return {"status": "online", "engine": "CargoComplianceEnv v1.0"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()