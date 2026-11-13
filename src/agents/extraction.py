"""Extraction & appraisal agent: full-text eligibility check, structured extraction with sources,
and risk-of-bias rating. Uses open-access full text when available (BM25 RAG), else the abstract.
Re-running on the same run folder only redoes failed records (or ones that now have full text)."""
import csv
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.llm import chat_json
from src.audit import log, save_json
from src.fileutil import safe_path
from src.tools.fulltext import fetch_fulltext
from src.rag.chunks import chunk_text, ChunkRetriever
from src.agents.screening import _criteria_text, norm_code, reason_label

ROB_ITEMS = ["selection_sampling", "method_description", "outcome_measurement",
             "analysis_validation", "reporting_completeness"]
RATINGS = {"low", "unclear", "high"}
MAX_CTX = 9000

FIELD_HINTS = {
    "study_design": "study design retrospective prospective randomized trial cohort experimental setup",
    "dataset": "dataset data collected cohort images patients source public private",
    "sample_size": "number of patients images samples participants total n =",
    "method": "method model architecture network approach proposed training",
    "metrics": "accuracy AUC sensitivity specificity F1 precision recall results performance",
    "main_findings": "results conclusion we found outperform significantly",
    "limitations": "limitation limitations future work weakness bias",
}

SYSTEM = (
    "You are a data-extraction reviewer in a PRISMA 2020 systematic review. Use ONLY the context given "
    "(the abstract and numbered excerpts [c#] from the full text, when present). Never guess: if something "
    'is not stated write "not reported". For each field give a short value (max 25 words) and "source": '
    '"abstract" or an excerpt id such as "c3". Also: (1) re-check eligibility against the numbered criteria '
    "(on the full-text excerpts when present, otherwise on the abstract; if unclear keep the record: include); "
    'give "reason_code" E# / E0 when excluding, else "none". (2) rate risk of bias for the 5 items below as '
    '"low", "unclear" or "high" with a one-sentence note; use "unclear" whenever the text does not say.\n'
    "ROB items: selection_sampling (representative, clearly described data/participants), method_description "
    "(method reproducible), outcome_measurement (appropriate reference standard / outcome), analysis_validation "
    "(proper train/test split, no leakage, external validation), reporting_completeness.\n"
    'Return JSON: {"eligibility":{"decision":"include|exclude","reason_code":"E1|E0|none","rationale":"..."},'
    '"fields":{"<field>":{"value":"...","source":"abstract|c#"}},'
    '"risk_of_bias":{"<item>":{"rating":"low|unclear|high","note":"..."}}}'
)


def build_context(rec, text, fields):
    parts = [f"[abstract] {(rec.get('abstract') or '')[:1800] or '(no abstract)'}"]
    if text:
        chunks = chunk_text(text)
        ret = ChunkRetriever(chunks)
        chosen = []
        for q in [FIELD_HINTS.get(f, f.replace("_", " ")) for f in fields] + \
                 ["eligibility criteria participants inclusion exclusion"]:
            for i in ret.top(q, 2):
                if i not in chosen:
                    chosen.append(i)
        parts += [f"[c{i}] {chunks[i]}" for i in sorted(chosen[:12])]
    return "\n\n".join(parts)[:MAX_CTX]


def rob_overall(rob):
    ratings = [v["rating"] for v in rob.values()]
    if "high" in ratings:
        return "high"
    if "unclear" in ratings:
        return "some concerns"
    return "low"


def extract_one(rec, text, protocol):
    fields = protocol["extraction_fields"]
    user = (_criteria_text(protocol) + "\n\nFIELDS TO EXTRACT: " + ", ".join(fields) +
            f"\n\nRECORD: {rec['title']} ({rec.get('year')})\n\nCONTEXT:\n" + build_context(rec, text, fields))
    d = chat_json(SYSTEM, user)
    n_exc = len(protocol["exclusion_criteria"])
    el = d.get("eligibility") or {}
    dec = "exclude" if str(el.get("decision", "")).lower().strip() == "exclude" else "include"
    code = norm_code(el.get("reason_code"), n_exc) if dec == "exclude" else "none"
    if dec == "exclude" and code == "none":
        code = "E0"
    got = d.get("fields") or {}
    out_fields = {}
    for f in fields:
        v = got.get(f) or {}
        if not isinstance(v, dict):
            v = {"value": str(v), "source": "?"}
        out_fields[f] = {"value": str(v.get("value") or "not reported"), "source": str(v.get("source") or "?")}
    rob_raw = d.get("risk_of_bias") or {}
    rob = {}
    for item in ROB_ITEMS:
        v = rob_raw.get(item) or {}
        rating = str(v.get("rating", "")).lower().strip() if isinstance(v, dict) else ""
        rob[item] = {"rating": rating if rating in RATINGS else "unclear",
                     "note": str(v.get("note", "")) if isinstance(v, dict) else ""}
    return {"eligibility": {"decision": dec, "reason_code": code,
                            "reason": reason_label(code, protocol), "rationale": str(el.get("rationale", ""))},
            "fields": out_fields, "risk_of_bias": rob, "overall_rob": rob_overall(rob)}


def fetch_all(recs, cache_dir):
    workers = int(os.getenv("FETCH_WORKERS", "6"))

    def one(r):
        try:
            return r["id"], fetch_fulltext(r, cache_dir)
        except Exception as e:
            return r["id"], (None, {"reason": f"error:{type(e).__name__}"})

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, recs))


def write_csv(path, rows, fields):
    cols = ["id", "title", "year", "doi", "text_source"] + fields + ROB_ITEMS + ["overall_rob"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["id"], r["title"], r.get("year"), r.get("doi", ""), r["text_source"]] +
                       [r["fields"][k]["value"] for k in fields] +
                       [r["risk_of_bias"][k]["rating"] for k in ROB_ITEMS] + [r["overall_rob"]])


def _failed_row(rec, p, text, info):
    return {"id": rec["id"], "title": rec["title"], "year": rec.get("year"), "doi": rec.get("doi", ""),
            "text_source": "full_text" if text else "abstract_only", "fulltext_info": info,
            "extraction_failed": True,
            "eligibility": {"decision": "include", "reason_code": "none", "reason": "extraction_error",
                            "rationale": "Extraction failed; kept pending a retry."},
            "fields": {f: {"value": "not reported", "source": "?"} for f in p["extraction_fields"]},
            "risk_of_bias": {k: {"rating": "unclear", "note": ""} for k in ROB_ITEMS},
            "overall_rob": "not assessed"}


def extraction(state):
    p, run_dir = state["protocol"], state["run_dir"]
    recs = state["included"]
    limit = int(os.getenv("EXTRACT_LIMIT", "0") or 0)
    if limit:
        recs = recs[:limit]
    texts = fetch_all(recs, os.path.join(run_dir, "fulltext"))
    n_full = sum(1 for t, _ in texts.values() if t)
    print(f"[extraction] full text retrieved for {n_full}/{len(recs)} records", flush=True)

    path = os.path.join(run_dir, "extraction.json")
    force = os.getenv("EXTRACT_FORCE", "0") == "1"
    prev = {}
    if os.path.exists(path):
        os.makedirs(os.path.join(run_dir, "backups"), exist_ok=True)
        shutil.copy2(path, os.path.join(run_dir, "backups", f"extraction_{time.strftime('%Y%m%d_%H%M%S')}.json"))
        try:
            with open(path, encoding="utf-8") as f:
                prev = {r["id"]: r for r in json.load(f)}
        except Exception:
            prev = {}

    rows, reused = [], 0
    for i, rec in enumerate(recs, 1):
        text, info = texts[rec["id"]]
        old = prev.get(rec["id"])
        # rows you corrected by hand ("curated": true) are never overwritten
        keep = old and (old.get("curated") or (not force and not old.get("extraction_failed")
                                               and not (old.get("text_source") == "abstract_only" and text)))
        if keep:
            rows.append(old)
            reused += 1
            continue
        try:
            res = extract_one(rec, text, p)
            row = {"id": rec["id"], "title": rec["title"], "year": rec.get("year"), "doi": rec.get("doi", ""),
                   "text_source": "full_text" if text else "abstract_only", "fulltext_info": info,
                   "extraction_failed": False, **res}
        except Exception as e:
            print(f"[extraction] failed for {rec['id']}: {str(e)[:200]}", flush=True)
            row = _failed_row(rec, p, text, info)
        rows.append(row)
        log(run_dir, "extraction", "record_processed", row)
        if i % 5 == 0 or i == len(recs):
            print(f"[extraction] {i}/{len(recs)} processed", flush=True)

    by_id = {r["id"]: r for r in recs}
    kept_all = [r for r in rows if r["eligibility"]["decision"] == "include"]
    kept_ok = [r for r in kept_all if not r["extraction_failed"]]
    final = [by_id[r["id"]] for r in kept_all]
    n_failed = sum(r["extraction_failed"] for r in rows)
    reasons = Counter(f"{r['eligibility']['reason_code']}: {r['eligibility']['reason'][:100]}"
                      for r in rows if r["eligibility"]["decision"] == "exclude")
    prisma = dict(state.get("prisma", {}))
    prisma.update({
        "fulltext_sought": len(recs),
        "fulltext_retrieved": n_full,
        "assessed_on_abstract_only": len(recs) - n_full,
        "excluded_after_fulltext_stage": len(rows) - len(kept_all),
        "exclusion_reasons_fulltext_stage": dict(reasons.most_common()),
        "included_in_review": len(kept_all),
        "extraction_pending_retry": n_failed,
        "risk_of_bias_overall": dict(Counter(r["overall_rob"] for r in kept_ok)),
        "extraction_reused_from_previous_run": reused,
    })
    prisma.pop("extraction_errors", None)
    save_json(run_dir, "extraction.json", rows)
    save_json(run_dir, "prisma_counts.json", prisma)
    log(run_dir, "extraction", "summary", prisma)
    try:
        write_csv(safe_path(os.path.join(run_dir, "evidence_table.csv")), kept_ok, p["extraction_fields"])
    except Exception as e:
        print(f"[extraction] could not write evidence_table.csv: {e}", flush=True)
    print(f"[extraction] {prisma}", flush=True)
    if n_failed:
        print(f"[extraction] {n_failed} records failed: run the same command again to retry only those.", flush=True)
    return {"extracted": rows, "final_included": final, "prisma": prisma}
