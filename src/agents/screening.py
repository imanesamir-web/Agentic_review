"""Title/abstract screening: two independent reviewers + arbiter for disagreements (PRISMA 2020).
Exclusion reasons are coded against the numbered protocol criteria (E1, E2, ...)."""
import os
import re
from collections import Counter

from src.llm import chat_json
from src.audit import log, save_json
from src.metrics import cohen_kappa

BATCH = 5
VALID = {"include", "exclude", "maybe"}

SYSTEM = (
    "You are a reviewer in a PRISMA 2020 systematic review doing TITLE/ABSTRACT screening. "
    "For each record decide: include, exclude, or maybe (not enough information in the abstract; "
    "it will go to full-text). Use ONLY the title/abstract and the numbered criteria given. "
    'If you exclude, set "reason_code" to the code of the main exclusion criterion that applies (e.g. "E2"); '
    'if none of the listed exclusion criteria fits, use "E0". Use "none" when including or unsure. '
    '"rationale" is one short sentence. '
    'Return {"decisions":[{"id":"1","decision":"include|exclude|maybe","reason_code":"E1|E0|none","rationale":"..."}]} '
    "with exactly one entry per record."
)

ARBITER_SYSTEM = (
    "You are the senior arbiter in a systematic review. Two reviewers disagreed on a record. "
    "Decide using the numbered criteria and the abstract. "
    'Return {"decision":"include|exclude","reason_code":"E1|E0|none","rationale":"one short sentence"}.'
)


def _criteria_text(p):
    inc = "\n".join(f"I{i + 1}: {c}" for i, c in enumerate(p["inclusion_criteria"]))
    exc = "\n".join(f"E{i + 1}: {c}" for i, c in enumerate(p["exclusion_criteria"]))
    return f"Review question: {p['research_question']}\nINCLUSION:\n{inc}\nEXCLUSION:\n{exc}"


def _rec_text(r, idx):
    abstract = (r.get("abstract") or "")[:1200] or "(no abstract)"
    return f"[{idx}] Title: {r['title']}\nYear: {r.get('year')}\nAbstract: {abstract}"


def norm_code(raw, n_exc):
    m = re.search(r"E\s*(\d+)", str(raw or ""), re.I)
    if not m:
        return "none"
    k = int(m.group(1))
    return f"E{k}" if 1 <= k <= n_exc else "E0"


def reason_label(code, protocol):
    if code.startswith("E") and code[1:].isdigit():
        k = int(code[1:])
        if 1 <= k <= len(protocol["exclusion_criteria"]):
            return protocol["exclusion_criteria"][k - 1]
        return "Other reason not covered by the listed criteria"
    return ""


def screen_batch(batch, protocol, model=None, temperature=0.0):
    """Returns {record_id: {decision, reason_code, reason, rationale}} for one batch."""
    idx_map = {str(i + 1): r for i, r in enumerate(batch)}
    n_exc = len(protocol["exclusion_criteria"])
    user = _criteria_text(protocol) + "\n\nRECORDS:\n" + "\n\n".join(
        _rec_text(r, i) for i, r in idx_map.items())
    out = {}
    try:
        data = chat_json(SYSTEM, user, model=model, temperature=temperature)
        for d in data.get("decisions", []):
            rec = idx_map.get(str(d.get("id")))
            if rec is None:
                continue
            dec = str(d.get("decision", "")).lower().strip()
            dec = dec if dec in VALID else "maybe"
            code = norm_code(d.get("reason_code"), n_exc) if dec == "exclude" else "none"
            if dec == "exclude" and code == "none":
                code = "E0"
            out[rec["id"]] = {"decision": dec, "reason_code": code,
                              "reason": reason_label(code, protocol),
                              "rationale": d.get("rationale", "")}
    except Exception as e:
        print(f"[screening] batch failed: {e}", flush=True)
    for r in batch:  # anything missing -> conservative 'maybe', flagged
        out.setdefault(r["id"], {"decision": "maybe", "reason_code": "none", "reason": "screening_error",
                                 "rationale": "No valid decision returned by the model."})
    return out


def arbitrate(rec, protocol, a, b):
    user = (_criteria_text(protocol) + "\n\n" + _rec_text(rec, 1) +
            f"\n\nReviewer A: {a['decision']} - {a['rationale']}\n"
            f"Reviewer B: {b['decision']} - {b['rationale']}")
    try:
        d = chat_json(ARBITER_SYSTEM, user)
        dec = str(d.get("decision", "")).lower().strip()
        if dec not in {"include", "exclude"}:
            dec = "include"  # conservative: send to full-text
        code = norm_code(d.get("reason_code"), len(protocol["exclusion_criteria"])) if dec == "exclude" else "none"
        if dec == "exclude" and code == "none":
            code = "E0"
        return {"decision": dec, "reason_code": code, "reason": reason_label(code, protocol),
                "rationale": d.get("rationale", "")}
    except Exception as e:
        print(f"[screening] arbiter failed: {e}", flush=True)
        return {"decision": "include", "reason_code": "none", "reason": "arbiter_error",
                "rationale": "Arbiter failed; kept for full-text."}


def _bin(d):  # 'maybe' goes to full-text, so it counts as include for agreement
    return "exclude" if d == "exclude" else "include"


def screening(state):
    p, run_dir = state["protocol"], state["run_dir"]
    recs = state["records"]
    limit = int(os.getenv("SCREEN_LIMIT", "0") or 0)
    if limit:
        recs = recs[:limit]
    model_b = os.getenv("SCREEN_MODEL_B") or None
    A, B = {}, {}
    for i in range(0, len(recs), BATCH):
        batch = recs[i:i + BATCH]
        A.update(screen_batch(batch, p, model=None, temperature=0.0))
        B.update(screen_batch(list(reversed(batch)), p, model=model_b, temperature=0.3))
        print(f"[screening] {min(i + BATCH, len(recs))}/{len(recs)} screened", flush=True)

    rows, included = [], []
    for r in recs:
        a, b = A[r["id"]], B[r["id"]]
        agree = _bin(a["decision"]) == _bin(b["decision"])
        if agree:
            src = a if a["decision"] == "exclude" else (b if b["decision"] == "exclude" else a)
            final = {"decision": _bin(a["decision"]), "reason_code": src["reason_code"] if _bin(a["decision"]) == "exclude" else "none",
                     "reason": src["reason"] if _bin(a["decision"]) == "exclude" else "",
                     "rationale": src["rationale"], "by": "agreement"}
        else:
            final = {**arbitrate(r, p, a, b), "by": "arbiter"}
        row = {"id": r["id"], "title": r["title"], "year": r.get("year"), "reviewer_a": a,
               "reviewer_b": b, "agreement": agree, "final": final}
        rows.append(row)
        log(run_dir, "screening_ta", "decision", row)
        if final["decision"] == "include":
            included.append(r)

    kappa = cohen_kappa([_bin(A[r["id"]]["decision"]) for r in recs],
                        [_bin(B[r["id"]]["decision"]) for r in recs])
    reasons = Counter(
        f"{x['final']['reason_code']}: {x['final']['reason'][:100]}"
        for x in rows if x["final"]["decision"] == "exclude")
    prisma = dict(state.get("prisma", {}))
    prisma.update({
        "screened_title_abstract": len(recs),
        "excluded_title_abstract": len(recs) - len(included),
        "exclusion_reasons": dict(reasons.most_common()),
        "disagreements_arbitrated": sum(not x["agreement"] for x in rows),
        "cohen_kappa": None if kappa is None else round(kappa, 3),
        "included_for_fulltext": len(included),
        "screening_errors": sum(1 for x in rows if "screening_error" in
                                (x["reviewer_a"]["reason"], x["reviewer_b"]["reason"])),
    })
    save_json(run_dir, "screening.json", rows)
    save_json(run_dir, "included_records.json", included)
    save_json(run_dir, "prisma_counts.json", prisma)
    log(run_dir, "screening_ta", "summary", prisma)
    print(f"[screening] {prisma}", flush=True)
    return {"screened": rows, "included": included, "prisma": prisma}
