"""Synthesis agent. Reads the curated extraction.json, groups studies (by task or imaging modality),
builds evidence tables, checks metric comparability (nothing is pooled), and writes a grounded narrative.
Grouping is cached in group_assignments.json: edit that file to correct it by hand, then re-run."""
import csv
import json
import os
from collections import Counter

from src.llm import chat, chat_json
from src.audit import log, save_json
from src.fileutil import safe_path
from src.synthesis_utils import (usable, field_value, row_rob, item_ratings, rob_counts, parse_metrics,
                                 eval_key, build_comparability, compact, has_value, ROB_ITEMS)

OTHER = "Other / unspecified"
GROUP_HINT = {
    "modality": "imaging modality (e.g. ultrasound, CT, MRI, X-ray, histopathology); use only if these are imaging studies",
    "task": "the task or clinical question addressed",
    "auto": "the task addressed or, for medical-imaging studies, the imaging modality, whichever gives clearer and "
            "more even groups",
}

SYS_LABELS = (
    "You organise the studies of a systematic review into groups for a narrative synthesis. Propose 3 to 7 "
    "mutually exclusive group labels (max 4 words each) based on the group_by instruction. Use 'Other / unspecified' "
    'only if really needed. Return {"groups":["label1","label2"]}.')
SYS_ASSIGN = (
    "Assign each study to exactly one of the given group labels, using only the study text. Also give the "
    "evaluation dataset used to report its results: a short canonical name if it is a NAMED dataset or benchmark "
    '(e.g. "BUSI"), otherwise "UNSPECIFIED" (private, in-house, institutional or not stated). Do not guess. '
    'Return {"assignments":[{"id":"1","group":"<label>","eval_set":"<name or UNSPECIFIED>"}]}.')
SYS_GROUP = (
    "You write the Results synthesis of a PRISMA 2020 systematic review (narrative synthesis, no meta-analysis). "
    "Use ONLY the study records provided. Every sentence that states a fact must end with citations like [S3] or "
    "[S3, S7] using the given study ids. NEVER pool, average or combine performance numbers across studies and never "
    "rank studies against each other; describe results study by study. Quote numbers exactly as written in the "
    "records. A numeric range may be given only if the FACTS list it as allowed. Cover: what was studied, methods, "
    "datasets and sample sizes when reported, the performance each study reports, heterogeneity, risk of bias "
    "(use the FACTS counts) and limitations. Write 'not reported' where data are missing. Do NOT generalise about the "
    "whole group ('the studies agree', 'no heterogeneity was reported'): make concrete statements about named studies "
    "and cite each one. 2 to 4 short paragraphs, "
    "plain text, no headings, no bullet points.")
SYS_SUMMARY = (
    "You write the 'Summary of findings' of a systematic review from the group passages provided. Only restate what "
    "the passages say; keep their [S#] citations; add no new facts or numbers beyond the FACTS. Do not pool or "
    "compare numbers across studies. Mention heterogeneity, risk of bias and gaps. 1 to 2 short paragraphs, plain text.")
SYS_LIMITS = (
    "You write the 'Limitations of the evidence' paragraph of a systematic review. Use ONLY the limitation notes and "
    "FACTS provided; cite studies as [S#]; do not invent limitations or numbers. Group recurring limitations, mention "
    "risk-of-bias patterns and the share of studies assessed on abstract only. 1 to 2 short paragraphs, plain text.")


def _env_int(name, default):
    return int(os.getenv(name, str(default)) or default)


def load_rows(state):
    path = os.path.join(state["run_dir"], "extraction.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return state.get("extracted", [])


def load_records(state):
    recs = {}
    for name in ("records.json", "included_records.json"):
        p = os.path.join(state["run_dir"], name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for r in json.load(f):
                    recs[r["id"]] = r
    for r in list(state.get("records", [])) + list(state.get("included", [])):
        recs.setdefault(r["id"], r)
    return recs


def build_studies(rows, recs):
    keep = sorted([r for r in rows if usable(r)], key=lambda r: (r.get("year") or 0, (r.get("title") or "").lower()))
    studies = []
    for i, r in enumerate(keep, 1):
        fields = {k: field_value(r, k) for k in (r.get("fields") or {})}
        metrics, amb = parse_metrics(fields.get("metrics", ""))
        rec = recs.get(r["id"], {})
        authors = rec.get("authors") or []
        studies.append({
            "sid": f"S{i:02d}", "id": r["id"], "title": r.get("title") or "", "year": r.get("year"),
            "doi": r.get("doi") or rec.get("doi", ""), "text_source": r.get("text_source", "abstract_only"),
            "fields": fields, "rob_overall": row_rob(r), "rob_items": item_ratings(r),
            "metrics": metrics, "metrics_ambiguous": amb, "group": OTHER, "eval_set": "UNSPECIFIED",
            "eval_key": None, "authors": authors, "venue": rec.get("venue") or "",
            "first_author": (authors[0].split()[-1] if authors else "")})
    return studies


# ---------- grouping ----------
def propose_labels(studies, protocol, group_by):
    lines = "\n".join(f"- {s['title'][:110]}" for s in studies[:120])
    user = (f"Review question: {protocol['research_question']}\ngroup_by: {GROUP_HINT[group_by]}\n"
            f"Studies:\n{lines}")
    try:
        labels = [str(x).strip() for x in chat_json(SYS_LABELS, user).get("groups", []) if str(x).strip()]
    except Exception as e:
        print(f"[synthesis] could not propose groups: {str(e)[:150]}", flush=True)
        labels = []
    seen, out = set(), []
    for lab in labels:
        if lab.casefold() not in seen and lab.casefold() != OTHER.casefold():
            seen.add(lab.casefold())
            out.append(lab)
    return out[:7] or ["All studies"]


def assign_batch(batch, labels, protocol):
    idx = {str(i + 1): s for i, s in enumerate(batch)}
    lines = "\n".join(
        f"[{i}] {s['title'][:120]} | dataset: {s['fields'].get('dataset', '')[:120]} | "
        f"method: {s['fields'].get('method', '')[:120]} | n: {s['fields'].get('sample_size', '')[:40]}"
        for i, s in idx.items())
    user = f"Group labels: {labels + [OTHER]}\nReview question: {protocol['research_question']}\nStudies:\n{lines}"
    out = {}
    try:
        for a in chat_json(SYS_ASSIGN, user).get("assignments", []):
            s = idx.get(str(a.get("id")))
            if s is not None:
                out[s["id"]] = {"group": str(a.get("group", "")), "eval_set": str(a.get("eval_set", "")) or "UNSPECIFIED"}
    except Exception as e:
        print(f"[synthesis] assignment batch failed: {str(e)[:150]}", flush=True)
    return out


def group_studies(studies, protocol, run_dir):
    """Reuse group_assignments.json if present (hand-editable); only missing studies go to the LLM."""
    path = os.path.join(run_dir, "group_assignments.json")
    saved = {}
    if os.path.exists(path) and os.getenv("SYNTH_REGROUP", "0") != "1":
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    labels = list(saved.get("labels", []))
    assign = {k: v for k, v in saved.get("assignments", {}).items()}
    missing = [s for s in studies if s["id"] not in assign]
    if missing:
        if not labels:
            labels = propose_labels(studies, protocol, os.getenv("SYNTH_GROUP_BY", "auto"))
        for i in range(0, len(missing), 8):
            assign.update({k: {**v, "title": next(s["title"] for s in missing if s["id"] == k)}
                           for k, v in assign_batch(missing[i:i + 8], labels, protocol).items()})
    lower = {lab.casefold(): lab for lab in labels}
    for s in studies:
        a = assign.get(s["id"], {})
        s["group"] = lower.get(str(a.get("group", "")).strip().casefold(), OTHER)
        s["eval_set"] = a.get("eval_set") or "UNSPECIFIED"
        s["eval_key"] = eval_key(s["eval_set"])
    save_json(run_dir, "group_assignments.json", {
        "labels": labels,
        "note": "Edit 'group' / 'eval_set' by hand to correct them, then re-run the synthesis stage.",
        "assignments": {s["id"]: {"title": s["title"], "group": s["group"], "eval_set": s["eval_set"]}
                        for s in studies}})
    return labels


# ---------- tables ----------
def _esc(x, n=70):
    x = " ".join(str(x).split())
    x = x if len(x) <= n else x[:n - 1] + "…"
    return x.replace("|", "/")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_esc(c, 90) for c in r) + " |" for r in rows]
    return "\n".join(out)


def group_table(studies):
    rows = []
    for s in studies:
        f = s["fields"]
        who = f"{s['first_author'] + ' et al.' if s['first_author'] else _esc(s['title'], 40)} {s['year']}"
        rows.append([s["sid"], who, f.get("dataset", ""), f.get("sample_size", ""), f.get("method", ""),
                     f.get("metrics", ""), s["rob_overall"], "full text" if s["text_source"] == "full_text" else "abstract"])
    return md_table(["ID", "Study", "Dataset", "n", "Method", "Metrics (as extracted)", "RoB", "Text"], rows)


def rob_table(groups):
    rows = []
    for label, sts in groups:
        c = rob_counts(sts)
        items = []
        for it in ROB_ITEMS:
            cc = Counter(s["rob_items"].get(it, "unclear") for s in sts)
            items.append(f"{cc['low']}/{cc['unclear']}/{cc['high']}")
        rows.append([label, len(sts), c["low"], c["some concerns"], c["high"], c["not assessed"], *items])
    heads = ["Group", "n", "Low", "Some concerns", "High", "Not assessed",
             *[f"{i.replace('_', ' ')} (L/U/H)" for i in ROB_ITEMS]]
    return md_table(heads, rows)


def comparability_table(comp):
    rows = []
    for c in comp:
        rng = f"{c['min']}–{c['max']}% on {c['eval_set']} (n={c['n_cluster']})" if c["range_allowed"] else "no (not comparable)"
        rows.append([c["group"], c["metric"], c["n_studies"], c["n_distinct_eval_sets"], rng,
                     c.get("note") or c.get("reason", "")])
    return md_table(["Group", "Metric", "Studies", "Named eval sets", "Descriptive range", "Note"], rows) if rows else \
        "No comparable numeric metrics could be parsed from the extracted data."


def write_table_csv(path, studies, field_names):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sid", "id", "group", "eval_set", "title", "year", "doi", "text_source", "rob_overall",
                    *field_names, "metrics_parsed"])
        for s in studies:
            w.writerow([s["sid"], s["id"], s["group"], s["eval_set"], s["title"], s["year"], s["doi"],
                        s["text_source"], s["rob_overall"], *[s["fields"].get(k, "") for k in field_names],
                        json.dumps(s["metrics"])])


# ---------- narrative ----------
def _clean(text):
    t = (text or "").strip()
    t = t.replace("```", "").strip()
    return "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#")).strip()


def group_facts(label, sts, comp):
    c = rob_counts(sts)
    full = sum(1 for s in sts if s["text_source"] == "full_text")
    years = [s["year"] for s in sts if s["year"]]
    ranges = [f"{x['metric']} on {x['eval_set']} (n={x['n_cluster']}): {x['min']}-{x['max']}% "
              f"[{', '.join(x['sids'])}] (descriptive only)" for x in comp if x["group"] == label and x["range_allowed"]]
    return (f"FACTS (you may use exactly these numbers):\n- Studies in group '{label}': {len(sts)}\n"
            f"- Overall risk of bias: low {c['low']}, some concerns {c['some concerns']}, high {c['high']}, "
            f"not assessed {c['not assessed']}\n- Assessed on full text: {full}; abstract only: {len(sts) - full}\n"
            f"- Publication years: {min(years) if years else 'n/a'}-{max(years) if years else 'n/a'}\n"
            f"- Allowed numeric ranges: {'; '.join(ranges) if ranges else 'none (metrics are not comparable across studies: do not compare numbers)'}")


def _generate(system, user, feedback=None, previous=None):
    if feedback:
        user += ("\n\nPREVIOUS DRAFT:\n" + (previous or "") +
                 "\n\nA reviewer found these problems. Rewrite the passage and fix every one "
                 "(remove or correct unsupported claims, numbers and citations):\n- " + "\n- ".join(feedback))
    try:
        return _clean(chat(system, user)), None
    except Exception as e:
        return "Narrative generation failed for this passage; see the evidence table.", str(e)[:200]


def _group_user(protocol, label, sts, facts):
    return (f"Review question: {protocol['research_question']}\nGroup: {label}\n\n{facts}\n\nSTUDIES:\n" +
            "\n".join(compact(s) for s in sts) + "\n\nWrite the passage.")


def _summary_user(protocol, syn):
    parts = []
    for g in syn["groups"]:
        txt = " ".join(b["text"] for b in syn["blocks"] if b["kind"] == "group" and b["group"] == g["label"])
        parts.append(f"## {g['label']} ({g['n']} studies)\n{txt[:1800]}")
    return f"Review question: {protocol['research_question']}\n\n{syn['facts_overall']}\n\n" + "\n\n".join(parts)


def _limits_user(protocol, syn):
    studies = syn["studies"]
    lines = [f"[{s['sid']}] {s['fields'].get('limitations', '')[:140]}" for s in studies
             if has_value(s["fields"].get("limitations", ""))][:40]
    items = []
    for it in ROB_ITEMS:
        cc = Counter(s["rob_items"].get(it, "unclear") for s in studies)
        items.append(f"{it.replace('_', ' ')}: low {cc['low']}, unclear {cc['unclear']}, high {cc['high']}")
    return (f"Review question: {protocol['research_question']}\n\n{syn['facts_overall']}\n"
            f"Risk-of-bias items across all studies: {'; '.join(items)}\n\nLIMITATION NOTES FROM STUDIES:\n" +
            ("\n".join(lines) or "(none reported)") + "\n\nWrite the paragraph.")


def overall_facts(studies, groups, n_rows_total):
    c = rob_counts(studies)
    full = sum(1 for s in studies if s["text_source"] == "full_text")
    years = [s["year"] for s in studies if s["year"]]
    return (f"FACTS (you may use exactly these numbers):\n- Studies synthesised: {len(studies)} (of {n_rows_total} curated rows)\n"
            f"- Groups: " + "; ".join(f"{g['label']} ({g['n']})" for g in groups) + "\n"
            f"- Overall risk of bias: low {c['low']}, some concerns {c['some concerns']}, high {c['high']}, "
            f"not assessed {c['not assessed']}\n- Assessed on full text: {full}; abstract only: {len(studies) - full}\n"
            f"- Publication years: {min(years) if years else 'n/a'}-{max(years) if years else 'n/a'}")


def _build_blocks(protocol, syn, part):
    blocks = []
    by_group = {g["label"]: [s for s in syn["studies"] if s["group"] == g["label"]] for g in syn["groups"]}
    for gi, g in enumerate(syn["groups"], 1):
        sts = by_group[g["label"]]
        for pi, i in enumerate(range(0, len(sts), part), 1):
            chunk = sts[i:i + part]
            facts = group_facts(g["label"], sts, syn["comparability"])
            text, err = _generate(SYS_GROUP, _group_user(protocol, g["label"], chunk, facts))
            blocks.append({"key": f"G{gi}P{pi}", "kind": "group", "group": g["label"],
                           "sids": [s["sid"] for s in chunk], "facts": facts, "text": text, "error": err})
            print(f"[synthesis] wrote {g['label']} part {pi}", flush=True)
    return blocks


def synthesis(state):
    p, run_dir = state["protocol"], state["run_dir"]
    feedback = state.get("critic_feedback") or {}
    part = _env_int("SYNTH_PART_SIZE", 12)

    if feedback and state.get("synthesis"):  # revision pass: rewrite only the flagged passages
        syn = state["synthesis"]
        by_sid = {s["sid"]: s for s in syn["studies"]}
        redo_groups = False
        for blk in syn["blocks"]:
            if blk["key"] in feedback and blk["kind"] == "group":
                sts = [by_sid[x] for x in blk["sids"]]
                user = _group_user(p, blk["group"], sts, blk["facts"])
                blk["text"], blk["error"] = _generate(SYS_GROUP, user, feedback[blk["key"]], blk["text"])
                redo_groups = True
        for blk in syn["blocks"]:
            if blk["kind"] == "summary" and (redo_groups or blk["key"] in feedback):
                blk["text"], blk["error"] = _generate(SYS_SUMMARY, _summary_user(p, syn),
                                                      feedback.get(blk["key"]) or ["Re-check against the revised group passages."],
                                                      blk["text"])
            if blk["kind"] == "limitations" and blk["key"] in feedback:
                blk["text"], blk["error"] = _generate(SYS_LIMITS, _limits_user(p, syn), feedback[blk["key"]], blk["text"])
        save_json(run_dir, "synthesis.json", syn)
        log(run_dir, "synthesis", "revised", {"blocks": sorted(feedback)})
        return {"synthesis": syn}

    rows = load_rows(state)
    studies = build_studies(rows, load_records(state))
    if not studies:
        raise RuntimeError("No usable studies in extraction.json (need eligibility 'include' and extracted data).")
    group_studies(studies, p, run_dir)
    if os.getenv("HUMAN_APPROVAL", "0") == "1":
        print("\n[human-in-the-loop] Studies grouped. Review / correct the groups in:\n  "
              + os.path.join(run_dir, "group_assignments.json"), flush=True)
        input("Edit 'group' / 'eval_set' if needed, save, then press Enter to write the narrative... ")
        group_studies(studies, p, run_dir)                 # re-reads the (edited) file
        log(run_dir, "synthesis", "grouping_approved", {})
    order = Counter(s["group"] for s in studies)
    labels = sorted(order, key=lambda g: (g == OTHER, -order[g], g))
    groups = [{"label": g, "n": order[g], "sids": [s["sid"] for s in studies if s["group"] == g]} for g in labels]
    comp = build_comparability(studies)
    syn = {"studies": studies, "groups": groups, "comparability": comp, "blocks": [],
           "facts_overall": overall_facts(studies, groups, len(rows)), "n_rows_total": len(rows),
           "n_include_rows": sum(1 for r in rows if (r.get("eligibility") or {}).get("decision") == "include")}
    syn["blocks"] = _build_blocks(p, syn, part)
    text, err = _generate(SYS_SUMMARY, _summary_user(p, syn))
    syn["blocks"].append({"key": "SUMMARY", "kind": "summary", "group": None, "sids": [s["sid"] for s in studies],
                          "facts": syn["facts_overall"], "text": text, "error": err})
    text, err = _generate(SYS_LIMITS, _limits_user(p, syn))
    syn["blocks"].append({"key": "LIMITATIONS", "kind": "limitations", "group": None,
                          "sids": [s["sid"] for s in studies], "facts": syn["facts_overall"], "text": text, "error": err})
    by_group = [(g["label"], [s for s in studies if s["group"] == g["label"]]) for g in groups]
    syn["tables"] = {"groups": {label: group_table(sts) for label, sts in by_group},
                     "rob": rob_table(by_group + [("All studies", studies)]),
                     "comparability": comparability_table(comp)}
    field_names = list(rows[0].get("fields", {}).keys()) if rows else []
    write_table_csv(safe_path(os.path.join(run_dir, "synthesis_table.csv")), studies, field_names)
    save_json(run_dir, "synthesis.json", syn)
    log(run_dir, "synthesis", "built", {"studies": len(studies), "groups": [(g["label"], g["n"]) for g in groups]})
    print(f"[synthesis] {len(studies)} studies in {len(groups)} groups; evidence tables written", flush=True)
    return {"synthesis": syn, "revision_count": state.get("revision_count", 0)}
