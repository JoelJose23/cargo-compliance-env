import random
from openenv.core.env_server import Environment
import uuid
from typing import Dict,List
from models import Cargo_Action,Cargo_Observation,Cargo_FetchState,Cargo_State
import json

# Mock Database for training 
GROUND_TRUTH = []

class CargoComplianceEnv(Environment):
    def __init__(self):
        # Dictionary to keep track of active sessions/agents
        self.session: Dict[str,Cargo_State] = {}
    
    def create_task(self):
        session_id = str(uuid.uuid4())
        task_id = random.choice(list(GROUND_TRUTH))