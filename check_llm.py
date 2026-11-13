"""Quick check of the configured LLM backend (.env). Usage:  py check_llm.py"""
import time

from src.llm import chat_json, model_name

t = time.time()
out = chat_json("You answer in JSON.", 'Return {"ok": true, "capital_of_france": "<fill in>"}')
print(f"model={model_name()}  answer={out}  ({time.time() - t:.1f}s)", flush=True)
