import argparse
import os
import time

from dotenv import load_dotenv

from src.graph import build_graph

load_dotenv()

ap = argparse.ArgumentParser()
ap.add_argument("question", help="Research question / topic of the systematic review")
args = ap.parse_args()

run_id = time.strftime("run_%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)
os.makedirs(run_dir, exist_ok=True)
print(f"[run] {run_id}", flush=True)

app = build_graph()
final = app.invoke({"run_id": run_id, "run_dir": run_dir, "question": args.question})
print("\nQueries:", final["queries"], flush=True)
print("PRISMA counts:", final["prisma"], flush=True)
print(f"Artifacts in {run_dir}/", flush=True)
