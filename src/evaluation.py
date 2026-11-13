"""Evaluation helpers: screening vs your own labels, extraction vs your curation, time vs a manual estimate."""
import csv
import json
import os
import random
import re
from collections import defaultdict

from src.fileutil import safe_path
from src.synthesis_utils import (field_value, item_ratings, row_rob, ROB_ITEMS, numeric_values, study_numbers,
                                 number_supported, numbers_in)


def _load(run_dir, name, default=None):
    p = os.path.join(run_dir, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _bin(d):
    return "exclude" if str(d).strip().lower() == "exclude" else "include"


def _div(a, b):
    return None if not b else a / b


def _r(x):
    return None if x is None else round(x, 3)


# ---------------------------------------------------------------- screening
def make_screening_sheet(run_dir, n=30, seed=42):
    """Blind labelling sheet (the agents' decisions are NOT shown). Balanced by the agents' final decision."""
    rows = _load(run_dir, "screening.json")
    if rows is None:
        raise FileNotFoundError(f"{run_dir}/screening.json not found")
    recs = {r["id"]: r for r in _load(run_dir, "records.json", [])}
    inc = [r for r in rows if r["final"]["decision"] == "include"]
    exc = [r for r in rows if r["final"]["decision"] == "exclude"]
    rng = random.Random(seed)
    k_exc = min(len(exc), n // 2)
    k_inc = min(len(inc), n - k_exc)
    k_exc = min(len(exc), n - k_inc)
    sample = rng.sample(inc, k_inc) + rng.sample(exc, k_exc)
    rng.shuffle(sample)
    out = os.path.join(run_dir, "eval")
    os.makedirs(out, exist_ok=True)
    sheet = safe_path(os.path.join(out, "screening_sheet.csv"))
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "year", "abstract", "human_decision", "notes"])
        for r in sample:
            w.writerow([r["id"], r["title"], r.get("year"), recs.get(r["id"], {}).get("abstract") or "", "", ""])
    key = {"seed": seed, "n_include_total": len(inc), "n_exclude_total": len(exc),
           "sampled_include": k_inc, "sampled_exclude": k_exc, "ids": [r["id"] for r in sample]}
    with open(os.path.join(out, "screening_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)
    proto = _load(run_dir, "protocol.json", {})
    with open(os.path.join(out, "criteria.txt"), "w", encoding="utf-8") as f:
        f.write("Label each record 'include' or 'exclude' (or 'maybe' = needs full text) using ONLY these criteria,\n"
                "WITHOUT looking at the pipeline's decisions.\n\nINCLUSION\n" +
                "\n".join(f"I{i + 1}: {c}" for i, c in enumerate(proto.get("inclusion_criteria", []))) +
                "\n\nEXCLUSION\n" +
                "\n".join(f"E{i + 1}: {c}" for i, c in enumerate(proto.get("exclusion_criteria", []))) + "\n")
    return sheet, key


def _metrics(tp, fp, tn, fn):
    sens, spec, prec = _div(tp, tp + fn), _div(tn, tn + fp), _div(tp, tp + fp)
    t = tp + fp + tn + fn
    f1 = None if (prec is None or sens is None or prec + sens == 0) else 2 * prec * sens / (prec + sens)
    kappa = None
    if t:
        po = (tp + tn) / t
        pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / t ** 2
        kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return {"sensitivity": _r(sens), "specificity": _r(spec), "precision": _r(prec),
            "accuracy": _r(_div(tp + tn, t)), "f1": _r(f1), "kappa": _r(kappa)}


def _interval_ci(rows, labels, predict, reps=4000, seed=7):
    """95% intervals that stay honest when a group has no errors in the sample. Records are grouped by (the pipeline's final
    decision, the predictor's decision); the unknown share q of human-eligible records in each group gets a Beta(k+.5, n-k+.5)
    distribution (Jeffreys prior) from the k of n labelled records, and the metrics are recomputed for each draw."""
    rng = random.Random(seed)
    size = defaultdict(int)
    for r in rows.values():
        size[(r["final"]["decision"], predict(r))] += 1
    n, k, ns, ks = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    for i, truth in labels.items():
        st = rows[i]["final"]["decision"]
        g = (st, predict(rows[i]))
        n[g] += 1
        ns[st] += 1
        if truth == "include":
            k[g] += 1
            ks[st] += 1
    samples = {m: [] for m in ("sensitivity", "specificity", "precision", "accuracy")}
    for _ in range(reps):
        tp = fp = tn = fn = 0.0
        for (st, pred), sz in size.items():
            g = (st, pred)
            if n[g]:
                a, b = k[g] + 0.5, n[g] - k[g] + 0.5
            else:                                   # no labelled record in this group: use its stratum
                a, b = ks[st] + 0.5, ns[st] - ks[st] + 0.5
            q = rng.betavariate(a, b)
            if pred == "include":
                tp, fp = tp + sz * q, fp + sz * (1 - q)
            else:
                fn, tn = fn + sz * q, tn + sz * (1 - q)
        m = _metrics(tp, fp, tn, fn)
        for key in samples:
            if m[key] is not None:
                samples[key].append(m[key])

    def pct(xs, q):
        xs = sorted(xs)
        return None if not xs else round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)

    return {"ci95": {m: [pct(v, 0.025), pct(v, 0.975)] for m, v in samples.items()},
            "ci_method": "Bayesian interval (Jeffreys prior) on the share of human-eligible records per group"}


def score_screening(run_dir):
    """Agents vs your labels. The sample is stratified by the agents' final decision, so each record is
    weighted by (stratum size / labelled in stratum) to estimate the metrics for the whole screened set."""
    rows = {r["id"]: r for r in _load(run_dir, "screening.json")}
    sheet = os.path.join(run_dir, "eval", "screening_sheet.csv")
    labels = {}
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("human_decision") or "").strip().lower()
            if d in ("include", "exclude", "maybe"):
                labels[r["id"]] = _bin(d)
    total = {"include": 0, "exclude": 0}
    for r in rows.values():
        total[r["final"]["decision"]] += 1
    n_lab = {"include": 0, "exclude": 0}
    for i in labels:
        n_lab[rows[i]["final"]["decision"]] += 1
    out = {"n_labelled": len(labels), "stratum_totals": total, "stratum_labelled": n_lab,
           "note": ("Weighted estimates for the whole screened set (stratified sample). With ~30 labelled records "
                    "the uncertainty is large: report the raw counts too."), "predictors": {}}
    preds = {"pipeline_final": lambda r: _bin(r["final"]["decision"]),
             "reviewer_a": lambda r: _bin(r["reviewer_a"]["decision"]),
             "reviewer_b": lambda r: _bin(r["reviewer_b"]["decision"])}
    for name, fn in preds.items():
        w = defaultdict(float)
        raw = defaultdict(int)
        for i, truth in labels.items():
            r = rows[i]
            stratum = r["final"]["decision"]
            wt = total[stratum] / n_lab[stratum]
            pred = fn(r)
            cell = ("tp" if pred == "include" else "fn") if truth == "include" else ("fp" if pred == "include" else "tn")
            w[cell] += wt
            raw[cell] += 1
        out["predictors"][name] = {"weighted": _metrics(w["tp"], w["fp"], w["tn"], w["fn"]),
                                   "raw_counts": {k: raw[k] for k in ("tp", "fp", "tn", "fn")},
                                   **_interval_ci(rows, labels, fn)}
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    with open(os.path.join(run_dir, "eval", "screening_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------- extraction
def original_extractions(run_dir):
    """The last successful LLM extraction per record, from the audit trail (before your manual corrections)."""
    path = os.path.join(run_dir, "decisions.jsonl")
    last = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                d = r.get("data") or {}
                if r.get("stage") == "extraction" and r.get("event") == "record_processed" \
                        and d.get("id") and not d.get("extraction_failed"):
                    last[d["id"]] = d
    return last


def _norm(x):
    v = re.sub(r"[\W_]+", " ", str(x or "").lower()).strip()
    return v or "not reported"


def extraction_changes(run_dir):
    """How much of the LLM extraction you had to change when curating extraction.json."""
    orig = original_extractions(run_dir)
    cur_rows = {r["id"]: r for r in _load(run_dir, "extraction.json", []) if not r.get("extraction_failed")}
    ids = [i for i in cur_rows if i in orig]
    per_field = defaultdict(lambda: [0, 0])
    by_source = defaultdict(lambda: [0, 0])
    rob_item = defaultdict(lambda: [0, 0])
    rob_overall = [0, 0]
    elig_changed = []
    for i in ids:
        o, c = orig[i], cur_rows[i]
        src = c.get("text_source", "?")
        for k in (c.get("fields") or {}):
            changed = _norm(field_value(o, k)) != _norm(field_value(c, k))
            per_field[k][0] += 1
            per_field[k][1] += changed
            by_source[src][0] += 1
            by_source[src][1] += changed
        io, ic = item_ratings(o), item_ratings(c)
        for it in ROB_ITEMS:
            if it in io or it in ic:
                rob_item[it][0] += 1
                rob_item[it][1] += io.get(it, "unclear") != ic.get(it, "unclear")
        rob_overall[0] += 1
        rob_overall[1] += row_rob(o) != row_rob(c)
        if (o.get("eligibility") or {}).get("decision") != (c.get("eligibility") or {}).get("decision"):
            elig_changed.append(i)
    n_f = sum(v[0] for v in per_field.values())
    n_c = sum(v[1] for v in per_field.values())
    res = {"records_compared": len(ids), "records_without_original": len(cur_rows) - len(ids),
           "field_edit_rate_overall": _r(_div(n_c, n_f)),
           "field_edit_rate_by_field": {k: {"n": v[0], "changed": v[1], "edit_rate": _r(_div(v[1], v[0]))}
                                        for k, v in sorted(per_field.items())},
           "field_edit_rate_by_text_source": {k: {"n": v[0], "changed": v[1], "edit_rate": _r(_div(v[1], v[0]))}
                                              for k, v in sorted(by_source.items())},
           "rob_item_agreement": {k: {"n": v[0], "agreement": _r(1 - v[1] / v[0])} for k, v in rob_item.items() if v[0]},
           "rob_overall_agreement": _r(1 - rob_overall[1] / rob_overall[0]) if rob_overall[0] else None,
           "eligibility_changed": elig_changed,
           "caveat": ("An edit rate only counts what you actually changed: it underestimates errors in records you did "
                      "not check, and cosmetic rewording counts as an edit. State how many records you reviewed.")}
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    with open(os.path.join(run_dir, "eval", "extraction_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


# ---------------------------------------------------------------- time
DEFAULT_ASSUMPTIONS = {
    "note": "Manual-effort figures are ASSUMPTIONS. Replace them with values you can cite or measure on a sample.",
    "minutes_per_abstract_per_reviewer": 1.0, "reviewers": 2, "minutes_per_conflict_resolution": 5.0,
    "minutes_per_study_fulltext_extraction_rob": 45.0, "hours_protocol_and_search_design": 4.0,
    "hours_synthesis_and_report": 12.0}


def time_report(run_dir):
    timing = _load(run_dir, "timing.json", {})
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    apath = os.path.join(run_dir, "eval", "time_assumptions.json")
    if not os.path.exists(apath):
        with open(apath, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_ASSUMPTIONS, f, indent=2)
    A = {**DEFAULT_ASSUMPTIONS, **_load(run_dir, "eval/time_assumptions.json", {})}
    pr = _load(run_dir, "prisma_counts.json", {})
    n_scr, n_dis, n_ext = pr.get("screened_title_abstract", 0), pr.get("disagreements_arbitrated", 0), \
        pr.get("fulltext_sought", 0)
    plan = [("Protocol, search and retrieval", ["planner", "search_strategy", "retrieval"],
             A["hours_protocol_and_search_design"] * 60),
            ("Title/abstract screening", ["screening"],
             n_scr * A["minutes_per_abstract_per_reviewer"] * A["reviewers"] + n_dis * A["minutes_per_conflict_resolution"]),
            ("Full text, extraction and risk of bias", ["extraction"], n_ext * A["minutes_per_study_fulltext_extraction_rob"]),
            ("Synthesis, quality review and report", ["synthesis", "critic", "report"], A["hours_synthesis_and_report"] * 60)]
    rows, tot_a, tot_m = [], 0.0, 0.0
    for label, stages, manual in plan:
        have = [timing[s]["seconds"] for s in stages if s in timing]
        auto = sum(have) / 60 if have else None
        runs = max([timing[s]["calls"] for s in stages if s in timing] or [0])
        rows.append({"stage": label, "runs": runs, "automated_min": None if auto is None else round(auto, 1),
                     "manual_estimate_min": round(manual, 1),
                     "speedup": None if not auto else round(manual / auto, 1)})
        if auto is not None:
            tot_a += auto
            tot_m += manual
    res = {"rows": rows, "total_timed_automated_min": round(tot_a, 1), "total_manual_estimate_min_same_stages": round(tot_m, 1),
           "speedup_timed_stages": None if not tot_a else round(tot_m / tot_a, 1), "assumptions": A,
           "note": "Stages without timing.json entries are not counted. Timing ADDS UP over re-runs of a stage: for a fair figure use a "
                   "run folder in which every stage ran once (a clean py run.py)."}
    with open(os.path.join(run_dir, "eval", "time_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


# ---------------------------------------------------------------- printing
def _fmt(x):
    return "n/a" if x is None else x


def format_screening(res):
    L = [f"Labelled: {res['n_labelled']}  (pipeline include/exclude in the whole set: "
         f"{res['stratum_totals']['include']}/{res['stratum_totals']['exclude']}; labelled: "
         f"{res['stratum_labelled']['include']}/{res['stratum_labelled']['exclude']})", "",
         "predictor        sens   spec   prec   acc    F1     kappa   raw (tp/fp/tn/fn)"]
    for name, v in res["predictors"].items():
        m, c = v["weighted"], v["raw_counts"]
        L.append(f"{name:<16} {_fmt(m['sensitivity']):<6} {_fmt(m['specificity']):<6} {_fmt(m['precision']):<6} "
                 f"{_fmt(m['accuracy']):<6} {_fmt(m['f1']):<6} {_fmt(m['kappa']):<7} "
                 f"{c['tp']}/{c['fp']}/{c['tn']}/{c['fn']}")
    L += ["", "95% intervals (Bayesian, Jeffreys prior) and the sample sizes behind them:"]
    for name, v in res["predictors"].items():
        c = v["ci95"]
        neg = v["raw_counts"]["tn"] + v["raw_counts"]["fp"]
        pos = v["raw_counts"]["tp"] + v["raw_counts"]["fn"]
        L.append(f"  {name:<15} sens {c['sensitivity']}  spec {c['specificity']}  prec {c['precision']}  acc {c['accuracy']}"
                 f"   (human-eligible labelled: {pos}, human-ineligible labelled: {neg})")
    return "\n".join(L + ["", res["note"]])


def format_extraction(res):
    L = [f"Records compared: {res['records_compared']} (without original LLM output: {res['records_without_original']})",
         f"Overall field edit rate: {_fmt(res['field_edit_rate_overall'])}",
         "By field: " + ", ".join(f"{k} {_fmt(v['edit_rate'])}" for k, v in res["field_edit_rate_by_field"].items()),
         "By text source: " + ", ".join(f"{k} {_fmt(v['edit_rate'])}" for k, v in res["field_edit_rate_by_text_source"].items()),
         "RoB item agreement: " + ", ".join(f"{k} {_fmt(v['agreement'])}" for k, v in res["rob_item_agreement"].items()),
         f"RoB overall agreement: {_fmt(res['rob_overall_agreement'])}",
         f"Eligibility decisions changed: {len(res['eligibility_changed'])}", "", res["caveat"]]
    return "\n".join(L)


def format_time(res):
    L = ["stage                                       automated   manual (assumed)   speed-up"]
    for r in res["rows"]:
        a = "not timed" if r["automated_min"] is None else f"{r['automated_min']} min" + (
            f" ({r['runs']} runs!)" if r.get("runs", 1) > 1 else "")
        L.append(f"{r['stage']:<43} {a:<11} {str(r['manual_estimate_min']) + ' min':<18} {_fmt(r['speedup'])}")
    L += ["", f"Timed stages total: {res['total_timed_automated_min']} min automated vs "
              f"{res['total_manual_estimate_min_same_stages']} min manual (assumed) = x{_fmt(res['speedup_timed_stages'])}",
          res["note"]]
    return "\n".join(L)


# ---------------------------------------------------------------- critic validation (seeded errors)
def critic_canary(run_dir):
    """Plants known errors in a REAL passage of your synthesis and checks whether the critic's deterministic checks
    catch them (nothing is written to the run folder). A faithful control sentence must raise no high-severity flag."""
    from src.agents import critic as cr
    syn = _load(run_dir, "synthesis.json")
    if syn is None:
        raise FileNotFoundError(f"{run_dir}/synthesis.json not found (run run_synthesis.py first)")
    studies = {s["sid"]: s for s in syn["studies"]}
    block = next(b for b in syn["blocks"] if b["kind"] == "group" and len(b["sids"]) >= 2)
    s1, s2 = block["sids"][0], block["sids"][1]
    facts = numeric_values(block.get("facts", ""))
    allowed1 = facts | study_numbers(studies[s1])

    def unsupported_number(candidates):
        for c in candidates:
            if not number_supported(float(c), len(c.split(".")[1]) if "." in c else 0, True, allowed1):
                return c
        return None

    fabricated = unsupported_number(["99.9", "87.3", "61.7", "73.4", "55.5"])
    # a number that really belongs to study 2 but not to study 1 (attribution error inside one paragraph)
    from_s2, attr_sid = None, None
    for cand in block["sids"][1:]:
        for v, d, pct, raw in numbers_in(" ".join(str(x) for x in studies[cand]["fields"].values())):
            if (d > 0 or v > 10) and not number_supported(v, d, pct, allowed1):
                from_s2, attr_sid = raw, cand
                break
        if from_s2:
            break
    other = next((s["sid"] for s in syn["studies"] if s["group"] != block["group"]), None)
    ghost = f"S{len(studies) + 50:02d}"
    metrics1 = studies[s1]["fields"].get("metrics", "") or "not reported"
    cases = [
        ("fabricated number", f"Study {s1} reported an accuracy of {fabricated}% [{s1}].", "number_unsupported"),
        ("pooling across studies", f"The average accuracy across these studies was high [{s1}, {s2}].", "pooling_language"),
        ("citation of a non-existent study", f"Another study confirmed this [{ghost}].", "unknown_citation"),
        ("overstated claim", f"This proves the method is definitive [{s1}].", "overclaim"),
    ]
    if other:
        cases.append(("citation from another group", f"The same pattern was seen elsewhere [{other}].", "wrong_group_citation"))
    if from_s2:
        cases.append(("number taken from another study in the same paragraph",
                      f"Study {s1} analysed {from_s2} cases [{s1}]. A different cohort was used in another study [{attr_sid}].",
                      "number_unsupported"))
    dataset1 = studies[s1]["fields"].get("dataset", "") or "its dataset"
    controls = {"faithful, cited": f"Study {s1} reported {metrics1} [{s1}].",
                "faithful, but citation only at the end of the paragraph":
                    f"Study {s1} reported {metrics1}. It was evaluated on {dataset1}. See the details [{s1}]."}
    scopes = ("paragraph", "inherit", "sentence")
    saved = os.environ.get("CRITIC_CITATION_SCOPE")
    rows, control_flags = [], {sc: {} for sc in scopes}
    try:
        for label, text, kind in cases:
            row = {"seeded_error": label, "expected_check": kind}
            for scope in scopes:
                os.environ["CRITIC_CITATION_SCOPE"] = scope
                found = cr.check_block({**block, "text": text, "error": None}, studies)
                row[scope] = any(i["kind"] == kind for i in found)
            rows.append(row)
        for scope in scopes:
            os.environ["CRITIC_CITATION_SCOPE"] = scope
            for label, text in controls.items():
                found = cr.check_block({**block, "text": text, "error": None}, studies)
                control_flags[scope][label] = sorted({i["kind"] for i in found if i["severity"] == "high"})
    finally:
        if saved is None:
            os.environ.pop("CRITIC_CITATION_SCOPE", None)
        else:
            os.environ["CRITIC_CITATION_SCOPE"] = saved
    res = {"block": block["key"], "cases": rows, "controls": control_flags,
           "detected": {sc: f"{sum(r[sc] for r in rows)}/{len(rows)}" for sc in scopes},
           "false_alarms_on_faithful_text": {sc: sum(1 for v in control_flags[sc].values() if v) for sc in scopes},
           "note": ("Deterministic checks only (the LLM check is not exercised). 'paragraph' accepts a number found in any "
                    "study cited in the same paragraph; 'sentence' also flags faithful sentences whose citation sits "
                    "elsewhere in the paragraph; 'inherit' is checked strictly when a sentence is cited and leniently "
                    "when it is not.")}
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    with open(os.path.join(run_dir, "eval", "critic_canary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def format_canary(res):
    L = [f"Seeded errors in passage {res['block']} (critic's deterministic checks)", "",
         "seeded error                                              expected check          paragraph  inherit    sentence"]
    for r in res["cases"]:
        L.append(f"{r['seeded_error']:<57} {r['expected_check']:<23} " +
                 "  ".join(f"{('caught' if r[sc] else 'MISSED'):<9}" for sc in ("paragraph", "inherit", "sentence")))
    L += ["", "Seeded errors caught:  " + ",  ".join(f"{sc} {res['detected'][sc]}" for sc in ("paragraph", "inherit", "sentence")),
          "Faithful sentences wrongly flagged (high severity):"]
    for sc in ("paragraph", "inherit", "sentence"):
        L.append(f"  {sc:<10} {res['false_alarms_on_faithful_text'][sc]} of {len(res['controls'][sc])}   "
                 + "; ".join(f"{k}: {v or 'ok'}" for k, v in res["controls"][sc].items()))
    return "\n".join(L + ["", res["note"]])


def critic_summary(run_dir, examples=2):
    """What is behind the critic's counts: findings by source, check and severity, with a few examples."""
    from collections import Counter
    rep = _load(run_dir, "critic_report.json")
    if rep is None:
        raise FileNotFoundError(f"{run_dir}/critic_report.json not found")
    issues = rep["issues"]
    by = Counter((i["severity"], i["source"], i["kind"]) for i in issues)
    order = {"high": 0, "medium": 1, "low": 2}
    table = [{"severity": k[0], "source": k[1], "check": k[2], "count": n}
             for k, n in sorted(by.items(), key=lambda kv: (order.get(kv[0][0], 3), -kv[1]))]
    ex = {}
    for i in issues:
        if i["severity"] == "high" and i.get("sentence"):
            ex.setdefault(i["kind"], [])
            if len(ex[i["kind"]]) < examples:
                ex[i["kind"]].append({"section": i["section"], "detail": i["detail"][:140], "sentence": i["sentence"][:200]})
    return {"config": rep.get("config"), "history": rep.get("history"), "verdict": rep.get("verdict"),
            "table": table, "high_examples": ex}


def format_critic_summary(res):
    L = [f"Verdict: {res['verdict']}   config: {res['config']}",
         "History: " + "; ".join(f"pass {h['pass']} {h['issues']}" + (f" [{h['note']}]" if h.get("note") else "")
                                 for h in (res["history"] or [])), "",
         "severity  source          check                      count"]
    for r in res["table"]:
        L.append(f"{r['severity']:<9} {r['source']:<15} {r['check']:<26} {r['count']}")
    for kind, items in res["high_examples"].items():
        L += ["", f"High-severity examples: {kind}"]
        L += [f"  [{e['section']}] {e['sentence']}\n      -> {e['detail']}" for e in items]
    return "\n".join(L)


def critic_review_sheet(run_dir, n=20, seed=42):
    """Sample of the critic's findings (LLM high / LLM medium / deterministic high) for you to judge: is the finding a
    REAL problem in the text, given the evidence shown? Gives the precision of each kind of finding."""
    from src.synthesis_utils import cited_sids, compact
    rep, syn = _load(run_dir, "critic_report.json"), _load(run_dir, "synthesis.json")
    if rep is None or syn is None:
        raise FileNotFoundError("critic_report.json / synthesis.json not found (run run_synthesis.py first)")
    studies = {s["sid"]: s for s in syn["studies"]}
    strata = {"llm/high": [], "llm/medium": [], "deterministic/high": []}
    for i, it in enumerate(rep["issues"]):
        k = f"{it['source']}/{it['severity']}"
        if k in strata and it.get("sentence"):
            strata[k].append(i)
    present = [k for k, v in strata.items() if v]
    rng = random.Random(seed)
    per = max(1, -(-n // max(len(present), 1)))
    chosen = []
    for k in present:
        chosen += [(k, i) for i in rng.sample(strata[k], min(per, len(strata[k])))]
    rng.shuffle(chosen)
    out = os.path.join(run_dir, "eval")
    os.makedirs(out, exist_ok=True)
    sheet = safe_path(os.path.join(out, "critic_review_sheet.csv"))
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "section", "sentence", "critic_finding", "evidence", "human_verdict (valid/invalid/unsure)", "notes"])
        for j, (k, i) in enumerate(chosen, 1):
            it = rep["issues"][i]
            sids = list(dict.fromkeys(cited_sids(it["sentence"])))
            blk = next((b for b in syn["blocks"] if b["key"] == it["section"]), {})
            sids = [x for x in sids if x in studies] or [x for x in blk.get("sids", []) if x in studies][:3]
            evidence = " || ".join(compact(studies[x], 700) for x in sids[:4])
            w.writerow([j, it["section"], it["sentence"], f"{it['kind']}: {it['detail']}", evidence, "", ""])
    with open(os.path.join(out, "critic_review_key.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "rows": {str(j): k for j, (k, _) in enumerate(chosen, 1)},
                   "stratum_sizes": {k: len(v) for k, v in strata.items()}}, f, indent=2)
    return sheet, {k: len(v) for k, v in strata.items()}


def score_critic_review(run_dir):
    key = _load(run_dir, "eval/critic_review_key.json")
    sheet = os.path.join(run_dir, "eval", "critic_review_sheet.csv")
    counts = defaultdict(lambda: {"valid": 0, "invalid": 0, "unsure": 0})
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("human_verdict (valid/invalid/unsure)") or "").strip().lower()
            if v in ("valid", "invalid", "unsure"):
                counts[key["rows"][str(r["id"])]][v] += 1
    out = {}
    for k, c in sorted(counts.items()):
        out[k] = {**c, "judged": c["valid"] + c["invalid"], "precision": _r(_div(c["valid"], c["valid"] + c["invalid"])),
                  "stratum_size": key["stratum_sizes"].get(k)}
    res = {"strata": out, "note": ("Precision = share of the critic's findings that you judged to be real problems. "
                                   "Small samples: report the counts, not only the percentage.")}
    with open(os.path.join(run_dir, "eval", "critic_review_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def format_critic_review(res):
    L = ["finding type          valid  invalid  unsure  precision  (findings of this type in the report)"]
    for k, v in res["strata"].items():
        L.append(f"{k:<21} {v['valid']:<6} {v['invalid']:<8} {v['unsure']:<7} {str(v['precision']):<10} {v['stratum_size']}")
    return "\n".join(L + ["", res["note"]])


def screening_disagreements(run_dir):
    """Records where the pipeline and your label differ, with the pipeline's reason: is it a model error or a
    difference in how the criteria were read?"""
    rows = {r["id"]: r for r in _load(run_dir, "screening.json")}
    out = []
    with open(os.path.join(run_dir, "eval", "screening_sheet.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("human_decision") or "").strip().lower()
            if d not in ("include", "exclude", "maybe") or r["id"] not in rows:
                continue
            row = rows[r["id"]]
            pred, human = _bin(row["final"]["decision"]), _bin(d)
            if pred != human:
                fin = row["final"]
                out.append({"id": r["id"], "title": r["title"], "pipeline": pred, "human": human,
                            "pipeline_reason": f"{fin.get('reason_code', '')}: {fin.get('reason', '')}".strip(": "),
                            "pipeline_rationale": fin.get("rationale", ""), "human_notes": r.get("notes", "")})
    return out


def format_disagreements(items):
    from collections import Counter
    if not items:
        return "No disagreements between the pipeline and your labels."
    c = Counter(i["pipeline_reason"] or "(none)" for i in items)
    L = [f"{len(items)} disagreements. Pipeline reasons: " + "; ".join(f"{k} x{v}" for k, v in c.most_common()), ""]
    for i in items:
        L += [f"- pipeline {i['pipeline']}, you {i['human']}: {i['title'][:110]}",
              f"    pipeline reason: {i['pipeline_reason']} | {i['pipeline_rationale'][:140]}"
              + (f" | your note: {i['human_notes']}" if i["human_notes"] else "")]
    return "\n".join(L)


VERIFY_FIELDS = ["study_design", "dataset", "sample_size", "method", "metrics"]
VERDICTS = {"correct": "correct", "c": "correct", "partial": "partial", "p": "partial", "wrong": "wrong", "w": "wrong",
            "unverifiable": "unverifiable", "u": "unverifiable"}


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def extraction_sheet(run_dir, n=12, seed=42, fields=None):
    """Verification sample: the extracted values next to the abstract; you mark each as correct / partial / wrong /
    unverifiable (not checkable without the full text). Gives the accuracy of the extraction, not just your edit rate."""
    fields = fields or VERIFY_FIELDS
    rows = [r for r in _load(run_dir, "extraction.json", []) if not r.get("extraction_failed")
            and (r.get("eligibility") or {}).get("decision") == "include"]
    if not rows:
        raise FileNotFoundError(f"{run_dir}/extraction.json has no extracted records")
    recs = {r["id"]: r for r in _load(run_dir, "records.json", [])}
    rng = random.Random(seed)
    full = [r for r in rows if r.get("text_source") == "full_text"]
    abst = [r for r in rows if r.get("text_source") != "full_text"]
    k_abs = min(len(abst), round(n * len(abst) / len(rows)))
    k_full = min(len(full), n - k_abs)
    sample = rng.sample(full, k_full) + rng.sample(abst, k_abs)
    rng.shuffle(sample)
    out = os.path.join(run_dir, "eval")
    os.makedirs(out, exist_ok=True)
    sheet = safe_path(os.path.join(out, "extraction_sheet.csv"))
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["record", "title", "doi", "text_source", "field", "llm_value", "verdict (correct/partial/wrong/unverifiable)",
                    "notes", "abstract (first row of each record)"])
        for j, r in enumerate(sample, 1):
            for k, fld in enumerate(fields):
                w.writerow([j, r["title"] if k == 0 else "", r.get("doi", "") if k == 0 else "",
                            r.get("text_source", "") if k == 0 else "", fld, field_value(r, fld), "", "",
                            (recs.get(r["id"], {}).get("abstract") or "") if k == 0 else ""])
    with open(os.path.join(out, "extraction_key.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "fields": fields,
                   "records": {str(j): {"id": r["id"], "text_source": r.get("text_source")} for j, r in enumerate(sample, 1)}},
                  f, indent=2)
    return sheet, {"records": len(sample), "full_text": k_full, "abstract_only": k_abs, "judgements": len(sample) * len(fields)}


def score_extraction_sheet(run_dir):
    key = _load(run_dir, "eval/extraction_key.json")
    sheet = os.path.join(run_dir, "eval", "extraction_sheet.csv")
    by_field = defaultdict(lambda: defaultdict(int))
    by_source = defaultdict(lambda: defaultdict(int))
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = VERDICTS.get((r.get("verdict (correct/partial/wrong/unverifiable)") or "").strip().lower())
            if not v:
                continue
            src = key["records"][str(r["record"])]["text_source"]
            by_field[r["field"]][v] += 1
            by_source[src][v] += 1

    def summ(c):
        judged = c["correct"] + c["partial"] + c["wrong"]
        return {"correct": c["correct"], "partial": c["partial"], "wrong": c["wrong"], "unverifiable": c["unverifiable"],
                "judged": judged, "accuracy_strict": _r(_div(c["correct"], judged)),
                "accuracy_strict_ci95": wilson(c["correct"], judged),
                "accuracy_lenient": _r(_div(c["correct"] + c["partial"], judged))}

    total = defaultdict(int)
    for c in by_field.values():
        for k, v in c.items():
            total[k] += v
    res = {"overall": summ(total), "by_field": {k: summ(v) for k, v in sorted(by_field.items())},
           "by_text_source": {k: summ(v) for k, v in sorted(by_source.items())},
           "note": ("strict = fully correct; lenient = correct or partially correct. 'unverifiable' judgements are not "
                    "counted. Report the number of records and judgements: the interval is wide for small samples.")}
    with open(os.path.join(run_dir, "eval", "extraction_verify.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def format_extraction_verify(res):
    def line(name, v):
        return (f"{name:<15} judged {v['judged']:<4} correct {v['correct']:<3} partial {v['partial']:<3} wrong {v['wrong']:<3} "
                f"unverifiable {v['unverifiable']:<3} strict {v['accuracy_strict']} {v['accuracy_strict_ci95']}  "
                f"lenient {v['accuracy_lenient']}")
    L = [line("OVERALL", res["overall"]), ""] + [line(k, v) for k, v in res["by_field"].items()] + [""] + \
        [line(k, v) for k, v in res["by_text_source"].items()]
    return "\n".join(L + ["", res["note"]])
