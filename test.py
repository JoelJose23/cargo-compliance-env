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
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Model did not return a JSON object. Raw output:\n{text}")
    return json.loads(match.group(0))


def parse_first_json_array(text: str) -> List[Any]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError(f"Model did not return a JSON array. Raw output:\n{text}")
    return json.loads(match.group(0))


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
    print("\n=== INITIAL OBSERVATION ===")
    print(obs.text)
    print("Manifest:", json.dumps(obs.manifest, indent=2))

    raw_text = obs.manifest.get("raw_text", "")

    extraction_prompt = (
        "Extract shipment details from the text. Return ONLY valid JSON with exactly these keys:\n"
        '{"qty": "...", "category"(Only choose from this list [Electronics,Food,Pharmaceutical,Nuclear]): "...", "Destination": "...", "Origin": "..."}\n'
        f"Text: {raw_text}"
    )
    extraction_raw = ollama_chat(
        model=model,
        messages=[{"role": "user", "content": extraction_prompt}],
        host=host,
        timeout=timeout,
    )
    extraction = parse_first_json_object(extraction_raw)
    print(f"DEBUG: LLM Extracted: {json.dumps(extraction, indent=2)}")

    obs = await env.step(
        session_id,
        Cargo_Action(
            action_type=Cargo_FetchState.VERDICT,
            decision=json.dumps(extraction),
        ),
    )
    print("\n=== AFTER EXTRACTION SUBMISSION ===")
    print(obs.text)
    print("Reward:", obs.reward)
    print("Current extraction:", json.dumps(obs.current_extraction, indent=2))

    if obs.available_laws:
        law_prompt = (
            "Pick the relevant law IDs for this shipment.\n"
            f"Extraction: {json.dumps(obs.current_extraction)}\n"
            f"Available laws: {json.dumps(obs.available_laws)}\n"
            "Return ONLY a JSON array of selected law IDs, e.g. [\"LAW_A\", \"LAW_B\"]."
        )
        law_raw = ollama_chat(
            model=model,
            messages=[{"role": "user", "content": law_prompt}],
            host=host,
            timeout=timeout,
        )
        selected_laws = parse_first_json_array(law_raw)

        obs = await env.step(
            session_id,
            Cargo_Action(
                action_type=Cargo_FetchState.PICK_LAW,
                decision=json.dumps(selected_laws),
            ),
        )
        print("\n=== AFTER LAW SELECTION ===")
        print(obs.text)
        print("Reward:", obs.reward)
        print("Selected laws:", json.dumps(selected_laws))
    else:
        print("\nNo laws available, extraction likely failed. Ending episode.")
        return

    reasoning_prompt = (
        "Write 3-5 concise sentences justifying selected laws for this shipment.\n"
        f"Extraction: {json.dumps(obs.current_extraction)}\n"
        f"Selected laws: {json.dumps(obs.current_extraction.get('laws', []))}"
    )
    reasoning = ollama_chat(
        model=model,
        messages=[{"role": "user", "content": reasoning_prompt}],
        host=host,
        timeout=timeout,
    )

    obs = await env.step(
        session_id,
        Cargo_Action(
            action_type=Cargo_FetchState.VERDICT,
            decision=reasoning,
        ),
    )
    print("\n=== FINAL VERDICT ===")
    print(obs.text)
    print("Reward:", obs.reward)
    print("Reasoning used:\n", reasoning)


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
