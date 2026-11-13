"""Run synthesis -> critic -> report on an existing run folder (uses the curated extraction.json as is).
Usage:  py run_synthesis.py runs\\run_XXXX            (writes the narrative, then checks it)
        py run_synthesis.py runs\\run_XXXX --reuse    (keeps the narrative in synthesis.json: only critic + report)"""
import json
import os
import shutil
import sys
import time

from dotenv import load_dotenv

load_dotenv()
from src.graph import build_synthesis_graph  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
reuse = "--reuse" in sys.argv
if len(args) != 1:
    sys.exit("Usage: py run_synthesis.py runs\\<run_folder> [--reuse]")
run_dir = args[0]


def load(name, default=None):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# keep a copy of everything this stage overwrites (the narrative can be rewritten by revision passes)
stamp = time.strftime("%Y%m%d_%H%M%S")
for name in ("synthesis.json", "critic_report.json", "final_report.md"):
    src = os.path.join(run_dir, name)
    if os.path.exists(src):
        os.makedirs(os.path.join(run_dir, "backups"), exist_ok=True)
        shutil.copy2(src, os.path.join(run_dir, "backups", f"{os.path.splitext(name)[0]}_{stamp}{os.path.splitext(name)[1]}"))

protocol = load("protocol.json")
if protocol is None:
    sys.exit(f"{run_dir}\\protocol.json not found")
state = {"run_id": os.path.basename(os.path.normpath(run_dir)), "run_dir": run_dir, "protocol": protocol,
         "question": protocol.get("research_question", ""), "prisma": load("prisma_counts.json", {}),
         "queries": load("queries.json", {}), "revision_count": 0}
entry = "synthesis"
if reuse:
    syn = load("synthesis.json")
    if syn is None:
        sys.exit("--reuse needs an existing synthesis.json (run once without --reuse)")
    state["synthesis"], entry = syn, "critic"
final = build_synthesis_graph(entry).invoke(state)
print(f"\nDone: {final['report_path']}  (critic verdict: {final['critic']['verdict']})", flush=True)
