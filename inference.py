import argparse
import asyncio
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List
from urllib import request


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow importing server/environment.py even when groq isn't installed.
if "groq" not in sys.modules:
    try:
        import groq  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        groq_stub = types.ModuleType("groq")

        class _FailingCompletions:
            @staticmethod
            def create(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("groq package is not installed.")

        class _FailingChat:
            completions = _FailingCompletions()

        class Groq:  # noqa: N801
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.chat = _FailingChat()

        groq_stub.Groq = Groq
        sys.modules["groq"] = groq_stub

# Allow importing server/environment.py even when tenacity isn't installed.
if "tenacity" not in sys.modules:
    try:
        import tenacity  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        tenacity_stub = types.ModuleType("tenacity")

        def retry(*args: Any, **kwargs: Any):
            def _decorator(fn):
                return fn
            return _decorator

        def stop_after_attempt(*args: Any, **kwargs: Any) -> Any:
            return None

        def wait_exponential(*args: Any, **kwargs: Any) -> Any:
            return None

        tenacity_stub.retry = retry
        tenacity_stub.stop_after_attempt = stop_after_attempt
        tenacity_stub.wait_exponential = wait_exponential
        sys.modules["tenacity"] = tenacity_stub

from models import Cargo_Action, Cargo_FetchState  # noqa: E402
from server.environment import CargoComplianceEnv  # noqa: E402


def ollama_chat(model: str, messages: List[Dict[str, str]], host: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        host,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["message"]["content"].strip()


def parse_first_json_object(text: str) -> Dict[str, Any]:
    # Use non-greedy match or finditer to get the first valid block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Fallback: try to find the last closing brace if the first match failed
        end_index = text.rfind("}") + 1
        start_index = text.find("{")
        return json.loads(text[start_index:end_index])

def parse_first_json_array(text: str) -> List[Any]:
    # Non-greedy match to avoid capturing trailing text/brackets
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in: {text}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        end_index = text.rfind(']') + 1
        start_index = text.find('[')
        return json.loads(text[start_index:end_index])


async def run_episode(model: str, host: str, timeout: int, use_ollama_judge: bool) -> None:
    env = CargoComplianceEnv()

    if use_ollama_judge:
        async def _local_judge(reasoning: str, extraction: dict, truth: dict) -> float:
            judge_prompt = (
                "Analyze the agent's compliance reasoning.\n"
                "Respond with exactly one line in this format: SCORE: [0.0-1.0]\n"
                f"Reasoning: {reasoning}"
            )
            raw = ollama_chat(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                host=host,
                timeout=timeout,
            )
            try:
                return max(0.0, min(1.0, float(re.findall(r"-?\d+(?:\.\d+)?", raw)[0])))
            except Exception:
                return 0.5

        env.get_llm_judge_score = _local_judge  # type: ignore[assignment]

    session_id, obs = env.create_task()
    cumulative_reward = 0.0
    print("\n=== INITIAL OBSERVATION ===")
    print(obs.text)
    
    raw_text = obs.manifest.get("raw_text", "")

    # CHANGE 1: Explicitly ask for units to prevent the string mismatch error 
    extraction_prompt = (
    "Extract shipment details. You MUST return a FLAT JSON object (no nesting). "
    "Use these exact keys: \"qty\", \"category\", \"Destination\", \"Origin\".\n"
    "For \"category\", choose one of: Food, Nuclear, Pharmaceutical, Electronics.\n"
    "Ensure 'qty' includes the units (e.g., '491 units').\n"
    f"Text: {raw_text}")
    
    extraction_raw = ollama_chat(model=model, messages=[{"role": "user", "content": extraction_prompt}], host=host, timeout=timeout)
    extraction = parse_first_json_object(extraction_raw)
    print(f"DEBUG: LLM Extracted: {json.dumps(extraction, indent=2)}")

    obs = await env.step(
        session_id,
        Cargo_Action(
            action_type=Cargo_FetchState.SUBMIT_EXTRACT,
            decision=json.dumps(extraction),
        ),
    )
    print("\n=== AFTER EXTRACTION SUBMISSION ===")
    print(obs.text)
    cumulative_reward += obs.reward
    print("Step Reward:", obs.reward)
    print("Total Reward:", obs.total_reward if hasattr(obs, "total_reward") else cumulative_reward)

    # CHANGE 2: Proper ID-to-Name Mapping
    # The environment rewards based on Law NAMES, but LLMs are more reliable with IDs.
    # We map the IDs back to Names before submitting the action.
    if obs.available_laws:
        law_prompt = (
        "SYSTEM: You are a Global Trade Custom Broker. You must provide a complete, bilateral compliance package JSON.\n"
        "USER:Create a compliance package for this shipment. You MUST select laws to clear BOTH the Origin (Export) and Destination (Import).\n"
        f"Shipment: {json.dumps(obs.current_extraction)}\n"
        f"Available Law IDs: {json.dumps(obs.available_laws)}\n\n"
        "Output ONLY this JSON format:\n"
        "{\n"
        "  \"laws\": [\"EXPORT_LAW_ID_HERE\", \"IMPORT_LAW_ID_HERE\"],\n"
        "  \"regulator\": \"Name of the specific regulator of BOTH countries (e.g., FDA, DEA, or CBP)\",\n"
        "  \"documents\": [\"List of required documents like Bill of Lading, Export License, etc.\"]\n"
        "}"
        )
        law_raw = ollama_chat(model=model, messages=[{"role": "user", "content": law_prompt}], host=host, timeout=timeout)
        # Parse the full compliance package.
        compliance_package = parse_first_json_object(law_raw)
        selected_ids = compliance_package.get("laws", [])
        print(f"DEBUG: LLM Selected Law IDs: {selected_ids}")

        # Map IDs back to Names for the environment's reward checker
        id_to_name_map = {law["id"]: law["name"] for law in obs.available_laws}
        valid_laws = []
        for l_id in selected_ids:
            if l_id in id_to_name_map:
                valid_laws.append(id_to_name_map[l_id])
            else:
                print(f"WARNING: LLM hallucinated an invalid Law ID: {l_id}")
                
        compliance_package["laws"] = valid_laws
        obs = await env.step(
            session_id,
            Cargo_Action(
                action_type=Cargo_FetchState.PICK_LAW,
                decision=json.dumps(compliance_package), # Environment receives the Names
            ),
        )
        print("\n=== AFTER LAW SELECTION ===")
        print(f"Status: {obs.text}")
        cumulative_reward += obs.reward
        print(f"Step Reward: {obs.reward}")
        print(f"Total Reward: {obs.total_reward if hasattr(obs, 'total_reward') else cumulative_reward}")
        print("Laws mapped and submitted:", compliance_package["laws"])
        print(f"DEBUG: Compliance Package being sent to Env: {json.dumps(compliance_package, indent=2)}")
    else:
        print("\nNo laws available. Check if extraction correctly identified the Category and Destination.")
        return

    # CHANGE 3: Better Reasoning Context
    # We now pass the names of the selected laws to the reasoning prompt.
    reasoning_prompt = (
        "Provide a 3-sentence legal justification for why this complete compliance package "
        "(laws, regulator, and documents) applies to this cargo.\n"
        f"Shipment Context: {json.dumps(obs.current_extraction)}\n"
        f"Selected Package: {json.dumps(compliance_package)}"
    )
    reasoning = ollama_chat(model=model, messages=[{"role": "user", "content": reasoning_prompt}], host=host, timeout=timeout)

    obs = await env.step(
        session_id,
        Cargo_Action(
            action_type=Cargo_FetchState.FINAL_VERDICT,
            decision=reasoning,
        ),
    )
    print("\n=== FINAL VERDICT ===")
    print(obs.text)
    cumulative_reward += obs.reward
    print("Step Reward:", obs.reward)
    print("Final Reward:", obs.total_reward if hasattr(obs, "total_reward") else cumulative_reward)
    print("Reasoning:\n", reasoning)

def main() -> None:
    parser = argparse.ArgumentParser(description="Test CargoComplianceEnv with a local Ollama model.")
    parser.add_argument("--model", default="mistral", help="Ollama model name.")
    parser.add_argument(
        "--host",
        default="http://localhost:11434/api/chat",
        help="Ollama chat endpoint.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--use-ollama-judge",
        action="store_true",
        help="Use Ollama instead of Groq for final audit scoring.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_episode(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            use_ollama_judge=args.use_ollama_judge,
        )
    )


if __name__ == "__main__":
    main()
