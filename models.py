from openenv.core.env_server import Action, Observation, State
from typing import List,Dict,Any,Optional
from enum import Enum

class Cargo_FetchState(str,Enum):
    FETCH_INFO = "fetch_info"
    PICK_LAW = "pick_law"
    VERDICT = "verdict"

class Cargo_Observation(Observation):
    text: str
    current_extraction: Optional[Dict[str,Any]] = None
    available_laws: List[Dict[str,str]] = []
    manifest: Dict[str, Any]
    laws: List[str]
    history: List[str]
    step: int
    reward: float = 0.0

class Cargo_State(State):
    task_id: str
    steps: int
    history: List[str]
    phase: str = "EXTRACTION"
    questions_asked: int = 0
    extraction_data: Optional[Dict[str,Any]] = None

class Cargo_Action(Action):
    action_type: Cargo_FetchState
    query: Optional[str] = None
    decision: Optional[str] = None
