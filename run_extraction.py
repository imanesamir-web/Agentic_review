"""Run ONLY the extraction stage on an existing run (no need to redo search and screening).
Rows you corrected by hand are protected if you add  "curated": true  to them. A backup of extraction.json
is saved in <run>\\backups before every run.
Usage:  py run_extraction.py runs\\run_XXXX"""
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()
from src.agents.extraction import extraction  # noqa: E402
from src.timing import timed  # noqa: E402

if len(sys.argv) != 2:
    sys.exit("Usage: py run_extraction.py runs\\<run_folder>")
run_dir = sys.argv[1]


def load(name):
    with open(os.path.join(run_dir, name), encoding="utf-8") as f:
        return json.load(f)


state = {"run_dir": run_dir, "protocol": load("protocol.json"),
         "included": load("included_records.json"), "prisma": load("prisma_counts.json")}
timed("extraction", extraction)(state)
print(f"Done. See {run_dir}\\evidence_table.csv and extraction.json", flush=True)
