import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# Setup project paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.models import Cargo_Action, Cargo_FetchState, Cargo_Observation

# --- MANDATORY HACKATHON CONFIGURATION ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "cargo-compliance")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "CargoComplianceEnv")
WORLD_ENV_URL = os.getenv("WORLD_ENV_URL", "https://SSETHackathonTeam-cargo-compliance-env.hf.space")
MAX_TOTAL_REWARD = float(os.getenv("MAX_TOTAL_REWARD", "5.5"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.44"))

EXTRACTION_FIELDS = ("qty", "category", "Destination", "Origin")


class CargoEnvClient:
    """HTTP client that connects the agent to the Cargo environment."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        print(f"🔗 Connected to World Environment at: {self.base_url}")

    def create_task(self) -> Tuple[Optional[str], Cargo_Observation]:
        """Create a session and return (session_id, observation).

        Supports both payload styles:
        1) {"session_id": "...", "observation": {...}}
        2) direct observation object (session is tracked server-side)
        """
        try:
            res = requests.post(f"{self.base_url}/reset", timeout=self.timeout)
            res.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"❌ Failed to connect to {self.base_url}. Ensure Docker/Uvicorn is running!"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"❌ /reset failed: {exc}") from exc

        data = res.json()
        if isinstance(data, dict) and "observation" in data:
            session_id = data.get("session_id")
            obs_payload = data["observation"]
        elif isinstance(data, dict):
            session_id = data.get("session_id")
            obs_payload = data
        else:
            raise RuntimeError(f"Unexpected /reset payload: {type(data).__name__}")

        if not isinstance(obs_payload, dict):
            raise RuntimeError("Observation payload from /reset is not a JSON object.")
        return session_id, Cargo_Observation(**obs_payload)

    async def step(self, session_id: Optional[str], action: Cargo_Action) -> Cargo_Observation:
        action_dict = action.model_dump() if hasattr(action, "model_dump") else action.dict()
        params = {"session_id": session_id} if session_id else None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(
                f"{self.base_url}/step",
                params=params,
                json=action_dict,
            )
            res.raise_for_status()
            data = res.json()

        if isinstance(data, dict) and "observation" in data:
            obs_dict = data["observation"] or {}
            obs_dict["reward"] = _to_float(data.get("reward"))
            obs_dict["total_reward"] = _to_float(data.get("total_reward"))
        elif isinstance(data, dict):
            obs_dict = data
            obs_dict.setdefault("reward", 0.0)
            obs_dict.setdefault("total_reward", 0.0)
        else:
            raise RuntimeError(f"Unexpected /step payload: {type(data).__name__}")

        if not isinstance(obs_dict, dict):
            raise RuntimeError("Observation payload from /step is not a JSON object.")
        return Cargo_Observation(**obs_dict)


# --- MANDATORY STDOUT LOGGERS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = f'"{error}"' if error else "null"
    safe_action = str(action).replace("\n", " ")
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from plain text / fenced code blocks."""
    if not text:
        return {}
    stripped = text.strip()

    candidates = [stripped]

    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, flags=re.IGNORECASE)
    candidates.extend(fence_matches)

    obj_match = re.search(r"\{[\s\S]*\}", stripped)
    if obj_match:
        candidates.append(obj_match.group(0))

    arr_match = re.search(r"\[[\s\S]*\]", stripped)
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return {}


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        for item in value:
            normalized = _normalize_scalar(item)
            if normalized:
                return normalized
        return ""
    if isinstance(value, dict):
        for key in ("value", "text", "name", "answer", "content"):
            normalized = _normalize_scalar(value.get(key))
            if normalized:
                return normalized
    return ""


def _merge_extraction(base: Dict[str, Any], update: Any) -> Dict[str, Any]:
    merged = dict(base or {})
    if not isinstance(update, dict):
        return merged

    for key in EXTRACTION_FIELDS:
        if key in update:
            val = _normalize_scalar(update.get(key))
            if val:
                merged[key] = val
    return merged


def _ensure_extraction_shape(extraction: Dict[str, Any]) -> Dict[str, str]:
    clean = {}
    for key in EXTRACTION_FIELDS:
        clean[key] = _normalize_scalar(extraction.get(key))
    return clean


def _normalize_documents(raw_documents: Any) -> List[str]:
    if raw_documents is None:
        return []
    if not isinstance(raw_documents, list):
        raw_documents = [raw_documents]

    docs: List[str] = []
    for item in raw_documents:
        value = _normalize_scalar(item)
        if value:
            docs.append(value)

    # Stable de-duplication.
    return list(dict.fromkeys(docs))


def _normalize_laws(raw_laws: Any, available_laws: List[Dict[str, Any]]) -> List[str]:
    """Map model output to known law names safely.

    Accepts entries like:
    - "LAW_001"
    - "Some Law Name"
    - {"id": "LAW_001"}
    - {"name": "Some Law Name"}
    """
    if raw_laws is None:
        return []
    if not isinstance(raw_laws, list):
        raw_laws = [raw_laws]

    id_to_name: Dict[str, str] = {}
    name_lookup: Dict[str, str] = {}
    for law in available_laws:
        law_id = _normalize_scalar(law.get("id"))
        law_name = _normalize_scalar(law.get("name"))
        if law_id and law_name:
            id_to_name[law_id] = law_name
        if law_name:
            name_lookup[law_name.lower()] = law_name

    selected: List[str] = []
    for item in raw_laws:
        candidates: List[str] = []
        if isinstance(item, str):
            candidates.append(item.strip())
        elif isinstance(item, dict):
            for key in ("id", "law_id", "name", "law", "title", "value"):
                if key in item:
                    normalized = _normalize_scalar(item.get(key))
                    if normalized:
                        candidates.append(normalized)
        else:
            normalized = _normalize_scalar(item)
            if normalized:
                candidates.append(normalized)

        for candidate in candidates:
            # Match exact ID.
            if candidate in id_to_name:
                selected.append(id_to_name[candidate])
                continue

            # Match ID token inside free text.
            id_match = re.search(r"\bLAW_\d+\b", candidate)
            if id_match and id_match.group(0) in id_to_name:
                selected.append(id_to_name[id_match.group(0)])
                continue

            # Match by name (case-insensitive).
            name_match = name_lookup.get(candidate.lower())
            if name_match:
                selected.append(name_match)

    return list(dict.fromkeys(selected))


def get_llm_response(client: OpenAI, sys_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception:
        return ""


async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    if not API_KEY:
        log_step(step=1, action="fail", reward=0.0, done=True, error="Missing HF_TOKEN/API_KEY")
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CargoEnvClient(base_url=WORLD_ENV_URL)

    try:
        session_id, obs = env.create_task()
        extraction_data: Dict[str, Any] = {}

        # --- PHASE 1: EXTRACTION ---
        raw_manifest = ""
        if isinstance(obs.manifest, dict):
            raw_manifest = str(obs.manifest.get("raw_text", ""))

        steps_taken += 1
        extraction_reply = get_llm_response(
            llm_client,
            'You are a logistics parser. Return JSON with keys: "qty", "category", "Destination", "Origin".',
            f"Extract details from shipment text:\n{raw_manifest}\nReturn JSON only.",
        )
        extraction_data = _merge_extraction(extraction_data, _extract_json(extraction_reply))
        extraction_data = _ensure_extraction_shape(extraction_data)

        obs = await env.step(
            session_id,
            Cargo_Action(
                action_type=Cargo_FetchState.SUBMIT_EXTRACT,
                decision=json.dumps(extraction_data, ensure_ascii=True),
            ),
        )

        reward = _to_float(obs.reward)
        rewards.append(reward)
        log_step(step=steps_taken, action="extract", reward=reward, done=False, error=None)

        # --- EXTRACTION REPAIR LOOP ---
        questions_used = 0
        max_extract_retries = 4
        extract_retries = 0

        while not obs.available_laws and extract_retries < max_extract_retries:
            extract_retries += 1

            decision_reply = get_llm_response(
                llm_client,
                (
                    "Output ONLY JSON: "
                    '{"action":"ask"|"submit","question":"...","extraction":{"qty":"...","category":"...","Destination":"...","Origin":"..."}}'
                ),
                f"Current extraction: {json.dumps(extraction_data)}\nEnvironment feedback: {obs.text}",
            )
            decision = _extract_json(decision_reply)
            action_type = _normalize_scalar(decision.get("action")).lower() if isinstance(decision, dict) else "submit"

            if action_type == "ask" and questions_used < 3:
                question = _normalize_scalar(decision.get("question")) if isinstance(decision, dict) else ""
                if not question:
                    question = "Could you confirm origin, destination, quantity, and category?"

                steps_taken += 1
                obs = await env.step(
                    session_id,
                    Cargo_Action(action_type=Cargo_FetchState.FETCH_INFO, decision=question),
                )
                reward = _to_float(obs.reward)
                rewards.append(reward)
                log_step(step=steps_taken, action="ask", reward=reward, done=False, error=None)
                questions_used += 1

                update_reply = get_llm_response(
                    llm_client,
                    'Output JSON object with keys: "qty", "category", "Destination", "Origin".',
                    f"Customer reply:\n{obs.text}",
                )
                extraction_data = _merge_extraction(extraction_data, _extract_json(update_reply))
            else:
                if isinstance(decision, dict):
                    extraction_data = _merge_extraction(extraction_data, decision.get("extraction", {}))

            extraction_data = _ensure_extraction_shape(extraction_data)

            steps_taken += 1
            obs = await env.step(
                session_id,
                Cargo_Action(
                    action_type=Cargo_FetchState.SUBMIT_EXTRACT,
                    decision=json.dumps(extraction_data, ensure_ascii=True),
                ),
            )
            reward = _to_float(obs.reward)
            rewards.append(reward)
            log_step(step=steps_taken, action="extract_retry", reward=reward, done=False, error=None)

            if "answered enough questions" in (obs.text or "").lower():
                break

        # --- PHASE 2: LAW SELECTION ---
        if obs.available_laws:
            steps_taken += 1
            law_reply = get_llm_response(
                llm_client,
                "Output ONLY JSON with keys: laws (array), regulator (string), documents (array).",
                (
                    f"Shipment extraction: {json.dumps(obs.current_extraction or {}, ensure_ascii=True)}\n"
                    f"Available law options: {json.dumps(obs.available_laws, ensure_ascii=True)}\n"
                    "Pick ONLY from available laws."
                ),
            )
            law_pkg = _extract_json(law_reply)
            if not isinstance(law_pkg, dict):
                law_pkg = {}

            package = {
                "laws": _normalize_laws(law_pkg.get("laws"), obs.available_laws or []),
                "regulator": _normalize_scalar(law_pkg.get("regulator")),
                "documents": _normalize_documents(law_pkg.get("documents")),
            }

            obs = await env.step(
                session_id,
                Cargo_Action(
                    action_type=Cargo_FetchState.PICK_LAW,
                    decision=json.dumps(package, ensure_ascii=True),
                ),
            )
            reward = _to_float(obs.reward)
            rewards.append(reward)
            log_step(step=steps_taken, action="laws", reward=reward, done=False, error=None)

            # --- PHASE 3: FINAL AUDIT ---
            steps_taken += 1
            reasoning = get_llm_response(
                llm_client,
                "Write exactly 3 concise sentences that justify compliance decisions.",
                (
                    f"Extraction: {json.dumps(obs.current_extraction or {}, ensure_ascii=True)}\n"
                    f"Selected package: {json.dumps(package, ensure_ascii=True)}"
                ),
                max_tokens=240,
            )
            if not reasoning:
                reasoning = (
                    "The selected laws align with the shipment origin and destination routes. "
                    "The regulator and documents were chosen to satisfy both export and import requirements. "
                    "This package reduces clearance risk by matching jurisdictional obligations."
                )

            obs = await env.step(
                session_id,
                Cargo_Action(action_type=Cargo_FetchState.FINAL_VERDICT, decision=reasoning),
            )
            reward = _to_float(obs.reward)
            rewards.append(reward)
            done = "episode finished" in (obs.text or "").lower()
            log_step(step=steps_taken, action="audit", reward=reward, done=done, error=None)

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        log_step(step=steps_taken + 1, action="fail", reward=0.0, done=True, error=_compact_error(exc))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
