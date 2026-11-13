from src.tools import openalex, pubmed
from src.tools.dedup import deduplicate
from src.audit import log, save_json


def retrieval(state):
    p, q, run_dir = state["protocol"], state["queries"], state["run_dir"]
    n = p["max_records_per_database"]
    yf, yt = p["date_range"]["from"], p["date_range"]["to"]
    raw, identified = [], {}
    if "pubmed" in p["databases"]:
        try:
            r = pubmed.search(q["pubmed"], n, yf, yt)
        except Exception as e:
            print(f"[retrieval] pubmed failed: {e}", flush=True)
            r = []
        identified["pubmed"] = len(r)
        raw += r
    if "openalex" in p["databases"]:
        try:
            r = openalex.search(q["openalex"], n, yf, yt)
        except Exception as e:
            print(f"[retrieval] openalex failed: {e}", flush=True)
            r = []
        identified["openalex"] = len(r)
        raw += r
    unique, removed = deduplicate(raw)
    prisma = {"identified": identified, "total_identified": len(raw),
              "duplicates_removed": len(removed), "records_after_dedup": len(unique)}
    save_json(run_dir, "records.json", unique)
    save_json(run_dir, "duplicates_removed.json", removed)
    save_json(run_dir, "prisma_counts.json", prisma)
    log(run_dir, "retrieval", "records_retrieved", prisma)
    print(f"[retrieval] {prisma}", flush=True)
    return {"records": unique, "prisma": prisma}
