"""Per-stage wall-clock timing, saved to <run_dir>/timing.json (used by the time-savings evaluation)."""
import json
import os
import time


def record(run_dir, stage, seconds):
    if not run_dir:
        return
    path = os.path.join(run_dir, "timing.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    e = data.setdefault(stage, {"seconds": 0.0, "calls": 0})
    e["seconds"] = round(e["seconds"] + seconds, 1)
    e["calls"] += 1
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def timed(name, fn):
    def wrapper(state):
        t0 = time.time()
        try:
            return fn(state)
        finally:
            record(state.get("run_dir"), name, time.time() - t0)

    wrapper.__name__ = name
    return wrapper
