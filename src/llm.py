"""LLM client over plain HTTP (OpenRouter / Ollama / any OpenAI-compatible server).
Falls back to other models on 429/5xx and puts a rate-limited model on a short cooldown.
Set LLM_JSON_MODE=1 to ask the server for JSON-only output (recommended with local Ollama models)."""
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

_cool = {}  # model -> timestamp until which it is skipped


def model_name() -> str:
    return os.getenv("LLM_MODEL", "")


def _models(model=None):
    fallbacks = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    first = model or model_name()
    return [first] + [m for m in fallbacks if m != first]


def _cooldown() -> float:
    return float(os.getenv("LLM_COOLDOWN_SECONDS", "45"))


def chat(system: str, user: str, temperature: float = 0.0, rounds: int = 4, model=None,
         json_mode: bool = False) -> str:
    base = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'ollama')}",
               "Content-Type": "application/json"}
    timeout = float(os.getenv("LLM_TIMEOUT", "300"))
    models = _models(model)
    last = None
    for rnd in range(rounds):
        now = time.time()
        avail = [m for m in models if _cool.get(m, 0) <= now]
        if not avail:  # everything is cooling down: wait for the soonest one
            soon = min(_cool[m] for m in models)
            time.sleep(min(max(soon - now, 1), 60))
            avail = models
        for m in avail:
            payload = {"model": m, "temperature": temperature,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}
            if json_mode and os.getenv("LLM_JSON_MODE", "0") == "1":
                payload["response_format"] = {"type": "json_object"}
            try:
                r = requests.post(f"{base}/chat/completions", headers=headers,
                                  json=payload, timeout=timeout)
            except Exception as e:
                last = f"{m}: {type(e).__name__}"
                _cool[m] = time.time() + 20
                print(f"[llm] {last} (cooling 20s)", flush=True)
                continue
            if r.status_code == 200:
                try:
                    content = r.json()["choices"][0]["message"]["content"]
                except Exception:
                    content = None
                if content:
                    return content
                last = f"{m}: empty answer"
                print(f"[llm] {last}", flush=True)
                continue
            last = f"{m}: HTTP {r.status_code} {r.text[:300]}"
            if r.status_code in (401, 402, 403):
                raise RuntimeError(f"LLM auth/credit error: {last}")
            if "free-models-per-day" in r.text:  # daily free quota used up: waiting will not help
                raise RuntimeError("OpenRouter daily free-model quota exhausted. Use a local Ollama model "
                                   "or add credits. " + last)
            if r.status_code in (429, 500, 502, 503, 504):
                _cool[m] = time.time() + _cooldown()
                print(f"[llm] {m}: HTTP {r.status_code}, cooling down {_cooldown():.0f}s", flush=True)
            else:
                print(f"[llm] {last}", flush=True)
        time.sleep(3)
    raise RuntimeError(f"LLM call failed: {last}")


def extract_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(text[start:end + 1])


def chat_json(system: str, user: str, retries: int = 3, temperature: float = 0.0, model=None) -> dict:
    system = system + "\nReturn ONLY a valid JSON object, no markdown, no commentary."
    last = None
    for _ in range(retries):
        out = chat(system, user, temperature=temperature, model=model, json_mode=True)
        try:
            return extract_json(out)
        except Exception as e:
            last = e
            user = user + "\n\nYour previous answer was not valid JSON. Return only JSON."
    raise RuntimeError(f"Could not parse JSON: {last}")
