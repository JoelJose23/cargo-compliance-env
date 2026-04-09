from openenv.core.env_server import Action, Observation, State
from typing import List, Dict, Any, Optional
from enum import Enum

class Cargo_FetchState(str, Enum):
    FETCH_INFO = "fetch_info" 
    PICK_LAW = "pick_law"     
    SUBMIT_EXTRACT = "submit_extract" 
    FINAL_VERDICT = "final_verdict"

class Cargo_Observation(Observation):
    text: str
    current_extraction: Optional[Dict[str, Any]] = None 
    available_laws: List[Dict[str, str]] = []
    available_documents: List[str] = []
    available_regulators: List[str] = []
    manifest: Dict[str, Any]
    laws: List[str]
    documents: List[str] = [] # To track selected paperwork
    regulator: Optional[str] = None #To track selected governing body
    duties: List[str] = [] # To track any calculated duties/tariffs
    history: List[str]
    step: int
    reward: float = 0.0
    total_reward: float = 0.0
    grader_score: Optional[float] = None

class Cargo_State(State):
    task_id: str
    steps: int
    history: List[str]
    phase: str = "EXTRACTION"
    questions_asked: int = 0
    total_reward: float = 0.0
    # Ensure this dictionary is initialized with the new keys in your environment.py
    extraction_data: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Dict[str, Any]] = None # Added for easier internal tracking

class Cargo_Action(Action):
    action_type: Cargo_FetchState
    query: Optional[str] = None
    # decision will now often be a JSON string containing multiple fields
    decision: Optional[str] = None
