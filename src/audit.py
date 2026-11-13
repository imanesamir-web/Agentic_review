"""Audit trail: every agent decision is appended to runs/<run_id>/decisions.jsonl."""
import hashlib
import json
import os
import time

from src.llm import model_name


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:12]


def log(run_dir: str, stage: str, event: str, data=None, prompt: str = ""):
    os.makedirs(run_dir, exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "event": event,
           "model": model_name(), "prompt_hash": prompt_hash(prompt) if prompt else None,
           "data": data}
    with open(os.path.join(run_dir, "decisions.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(run_dir: str, name: str, obj):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
