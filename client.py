# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cargo Compliance Environment Client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import Cargo_Action, Cargo_Observation


class CargoComplianceEnvClient(
    EnvClient[Cargo_Action, Cargo_Observation, State]
):
    """
    Client for the Cargo Compliance Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with CargoComplianceEnvClient(base_url="http://localhost:7860") as client:
        ...     result = client.reset(task_id="cargo_food")
        ...     print(result.observation.text)
        ...
        ...     from models import Cargo_FetchState
        ...     action = Cargo_Action(
        ...         action_type=Cargo_FetchState.SUBMIT_EXTRACT,
        ...         decision='{"qty":"100 units","category":"Food","Origin":"India","Destination":"United States"}'
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.text)
    """

    def _step_payload(self, action: Cargo_Action) -> Dict:
        """
        Convert Cargo_Action to JSON payload for step message.

        Args:
            action: Cargo_Action instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value if hasattr(action.action_type, "value") else action.action_type,
            "decision": action.decision,
            "query": action.query if hasattr(action, "query") else None,
        }

    def _parse_result(self, payload: Dict) -> StepResult[Cargo_Observation]:
        """
        Parse server response into StepResult[Cargo_Observation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with Cargo_Observation
        """
        obs_data = payload.get("observation", payload)

        observation = Cargo_Observation(
            text=obs_data.get("text", ""),
            current_extraction=obs_data.get("current_extraction"),
            available_laws=obs_data.get("available_laws", []),
            available_documents=obs_data.get("available_documents", []),
            available_regulators=obs_data.get("available_regulators", []),
            manifest=obs_data.get("manifest", {}),
            laws=obs_data.get("laws", []),
            documents=obs_data.get("documents", []),
            regulator=obs_data.get("regulator"),
            duties=obs_data.get("duties", []),
            history=obs_data.get("history", []),
            step=obs_data.get("step", 0),
            reward=float(obs_data.get("reward", 0.0)),
            total_reward=float(obs_data.get("total_reward", 0.0)),
            grader_score=obs_data.get("grader_score"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object
        """
        return State(
            episode_id=payload.get("episode_id") or payload.get("task_id"),
            step_count=payload.get("step_count") or payload.get("steps", 0),
        )