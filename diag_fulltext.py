"""Shows, per record, why full text was or was not found.  Usage:  py diag_fulltext.py runs\\run_XXXX [N]"""
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()
from src.tools import fulltext as ft  # noqa: E402

run_dir = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
with open(os.path.join(run_dir, "included_records.json"), encoding="utf-8") as f:
    recs = json.load(f)[:n]
for r in recs:
    print("-", r["id"], "| doi:", r.get("doi") or "-", flush=True)
    print("    openalex pdf:", r.get("oa_pdf") or "-")
    t, s = ft.epmc_fulltext(r)
    print("    europepmc   :", f"OK {len(t)} chars" if t else s)
    print("    unpaywall   :", ft.unpaywall_pdf(r.get("doi", "")) or "-")
    t, info = ft.fetch_fulltext(r, os.path.join(run_dir, "_diag_cache"))
    print("    RESULT      :", f"OK {len(t)} chars via {info.get('source_url')}" if t else info, flush=True)
