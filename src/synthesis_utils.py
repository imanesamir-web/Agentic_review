"""Pure helpers shared by the synthesis, critic and report stages (no LLM calls)."""
import re
from collections import Counter, defaultdict

from src.agents.extraction import rob_overall, ROB_ITEMS

NO_VALUE = {"", "not reported", "n/a", "na", "none", "unclear", "?", "not stated", "unknown"}
OVERALL = ("low", "some concerns", "high")
CITE_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


# ---------- reading curated rows ----------
def has_value(v) -> bool:
    return str(v or "").strip().lower() not in NO_VALUE


def field_value(row, name) -> str:
    f = (row.get("fields") or {}).get(name)
    if isinstance(f, dict):
        return str(f.get("value", "") or "")
    return str(f or "")


def usable(row) -> bool:
    """Eligible rows with usable data. A row still flagged as failed counts only if it was curated by hand."""
    if (row.get("eligibility") or {}).get("decision") != "include":
        return False
    if not row.get("extraction_failed"):
        return True
    return any(has_value(field_value(row, k)) for k in (row.get("fields") or {}))


def item_ratings(row) -> dict:
    out = {}
    for item, v in (row.get("risk_of_bias") or {}).items():
        r = (v or {}).get("rating") if isinstance(v, dict) else v
        out[item] = r if r in ("low", "unclear", "high") else "unclear"
    return out


def computed_rob(row):
    items = item_ratings(row)
    return rob_overall({k: {"rating": r} for k, r in items.items()}) if items else None


def row_rob(row) -> str:
    o = row.get("overall_rob")
    if o in OVERALL:
        return o
    return computed_rob(row) or "not assessed"


def rob_inconsistent(row):
    """(stored, computed) if a curated overall rating disagrees with its item ratings, else None."""
    stored, comp = row.get("overall_rob"), computed_rob(row)
    if stored in OVERALL and comp and stored != comp:
        return stored, comp
    return None


def rob_counts(studies) -> dict:
    c = Counter(s["rob_overall"] for s in studies)
    return {k: c.get(k, 0) for k in (*OVERALL, "not assessed")}


# ---------- metric parsing and comparability ----------
_ALIAS = {"accuracy": "Accuracy", "acc": "Accuracy", "auc": "AUC", "auroc": "AUC", "roc auc": "AUC",
          "auc roc": "AUC", "sensitivity": "Sensitivity", "specificity": "Specificity", "recall": "Recall",
          "precision": "Precision", "f1": "F1", "f1 score": "F1", "dice": "Dice", "iou": "IoU", "kappa": "Kappa"}
_NAME = r"(accuracy|acc|auroc|roc[- ]auc|auc[- ]roc|auc|sensitivity|specificity|recall|precision|f1[- ]score|f1|dice|iou|kappa)"
_NUM = r"(\d{1,3}(?:\.\d+)?)\s*(%?)"
_GAP = r"\s*(?:(?:=|:|≈|~|\(|of|was|is|reached|achieved|scores?|up to|around|approximately)\s*)*"
_P1 = re.compile(r"\b" + _NAME + r"\b" + _GAP + _NUM, re.I)
_P2 = re.compile(_NUM + r"\s*(?:of\s+)?" + _NAME + r"\b", re.I)


def _norm_metric(name):
    return _ALIAS.get(re.sub(r"[- ]+", " ", name.lower()).strip())


def _value(metric, num, pct):
    v = float(num)
    if metric == "Kappa":
        return round(v, 3) if v <= 1 else None
    if pct or v > 1:
        return round(v, 1) if v <= 100 else None
    return round(v * 100, 1)


def parse_metrics(text):
    """Returns ({metric: value in %}, [ambiguous metrics]). A metric reported with several different
    values in one study is ambiguous and is never used for comparisons."""
    found = defaultdict(set)
    for m in _P1.finditer(text or ""):
        name = _norm_metric(m.group(1))
        val = _value(name, m.group(2), m.group(3)) if name else None
        if val is not None:
            found[name].add(val)
    for m in _P2.finditer(text or ""):
        name = _norm_metric(m.group(3))
        val = _value(name, m.group(1), m.group(2)) if name else None
        if val is not None:
            found[name].add(val)
    metrics = {k: next(iter(v)) for k, v in found.items() if len(v) == 1}
    return metrics, sorted(k for k, v in found.items() if len(v) > 1)


_UNSPEC = ("private", "in-house", "inhouse", "institution", "proprietary", "unspecified", "not reported",
           "unknown", "custom", "own dataset", "collected", "hospital", "single-center", "single center",
           "multi-center", "multicenter", "local")


def eval_key(name):
    n = (name or "").strip().lower()
    if not n or any(w in n for w in _UNSPEC):
        return None
    return re.sub(r"[^a-z0-9]", "", n) or None


def build_comparability(studies, min_n=3):
    """Per (group, metric): may a descriptive range be shown? Only when >= min_n studies report the
    same metric on the same NAMED evaluation dataset. Nothing is ever pooled or averaged."""
    by = defaultdict(list)
    for s in studies:
        for m, v in s["metrics"].items():
            by[(s["group"], m)].append((s["sid"], v, s.get("eval_key"), s.get("eval_set")))
    out = []
    for (g, m), items in sorted(by.items()):
        keys = Counter(k for _, _, k, _ in items if k)
        entry = {"group": g, "metric": m, "n_studies": len(items),
                 "n_unspecified_eval_set": sum(1 for it in items if not it[2]),
                 "n_distinct_eval_sets": len(keys), "range_allowed": False}
        if keys:
            best, best_n = keys.most_common(1)[0]
            if best_n >= min_n:
                cl = [it for it in items if it[2] == best]
                vals = [it[1] for it in cl]
                entry.update(range_allowed=True, eval_set=cl[0][3], n_cluster=best_n,
                             min=min(vals), max=max(vals), sids=[it[0] for it in cl],
                             note="Descriptive range only; data splits and protocols may still differ.")
        if not entry["range_allowed"]:
            entry["reason"] = (f"fewer than {min_n} studies report {m} on the same named evaluation dataset "
                               f"({len(keys)} distinct named sets, {entry['n_unspecified_eval_set']} unspecified/private)")
        out.append(entry)
    return out


# ---------- text helpers (citations, sentences, numbers) ----------
def cited_sids(text):
    out = []
    for m in CITE_RE.finditer(text or ""):
        out += [x.strip() for x in m.group(1).split(",")]
    return out


def strip_cites(text):
    return CITE_RE.sub("", text or "")


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%)?")


def numbers_in(text):
    """[(value, decimals, is_percent, raw)] for the numbers written in a text."""
    out = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(1)
        out.append((float(raw), len(raw.split(".")[1]) if "." in raw else 0, bool(m.group(2)), raw))
    return out


def numeric_values(text):
    vals = set()
    for v, _, _, _ in numbers_in(text):
        vals.update({v, v / 100, v * 100})
    return vals


def number_supported(x, decimals, is_pct, allowed) -> bool:
    if decimals == 0 and 1900 <= x <= 2100:
        return True
    if decimals == 0 and x <= 10 and not is_pct:
        return True
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(v - x) <= tol for v in allowed)


def study_numbers(study):
    text = " ".join(str(v) for v in study["fields"].values()) + f" {study.get('year') or ''}"
    return numeric_values(text)


POOL_RE = re.compile(
    r"\b(pooled?|pooling|meta-analy\w+|average[sd]?|mean (?:accuracy|auc|sensitivity|specificity|performance|f1)|"
    r"on average|weighted (?:mean|average)|overall (?:accuracy|auc|sensitivity|specificity|performance) of|"
    r"combined (?:accuracy|auc|sensitivity))\b", re.I)
NEGATION_RE = re.compile(r"\b(no|not|without|cannot|could not|unable|avoid\w*|nor)\b", re.I)
OVERCLAIM_RE = re.compile(r"\b(prove[sd]?|proven|definitive(?:ly)?|best[- ]performing|clearly superior|"
                          r"outperforms? all|guarantee[sd]?)\b", re.I)


# ---------- compact study view for prompts ----------
def compact(s, n=170):
    f = s["fields"]

    def t(k):
        v = f.get(k, "")
        return (v if has_value(v) else "not reported")[:n]

    return (f"[{s['sid']}] {s['title'][:140]} ({s['year']}) | design: {t('study_design')} | dataset: {t('dataset')} | "
            f"n: {t('sample_size')} | method: {t('method')} | metrics: {t('metrics')} | "
            f"findings: {t('main_findings')} | limitations: {t('limitations')} | RoB: {s['rob_overall']} | "
            f"text: {s['text_source']}")


# ---------- PRISMA arithmetic ----------
def prisma_checks(p, n_include_rows):
    """Consistency checks on the PRISMA counts. Returns [(kind, severity, detail)]."""
    out = []

    def g(k):
        return p.get(k)

    def need(cond, detail, sev="medium"):
        if cond is False:
            out.append(("prisma_arithmetic", sev, detail))

    if None not in (g("total_identified"), g("duplicates_removed"), g("records_after_dedup")):
        need(g("total_identified") - g("duplicates_removed") == g("records_after_dedup"),
             "identified - duplicates != records after deduplication")
    if None not in (g("screened_title_abstract"), g("excluded_title_abstract"), g("included_for_fulltext")):
        need(g("screened_title_abstract") - g("excluded_title_abstract") == g("included_for_fulltext"),
             "screened - excluded (title/abstract) != included for full-text stage")
    if None not in (g("included_for_fulltext"), g("fulltext_sought")):
        need(g("included_for_fulltext") == g("fulltext_sought"),
             "included after screening != reports sought (extraction was limited or run on a subset)")
    if None not in (g("fulltext_sought"), g("excluded_after_fulltext_stage"), g("included_in_review")):
        need(g("fulltext_sought") - g("excluded_after_fulltext_stage") == g("included_in_review"),
             "reports sought - excluded != included in review")
    if g("included_in_review") is not None:
        need(g("included_in_review") == n_include_rows,
             f"PRISMA 'included in review' ({g('included_in_review')}) differs from the curated rows "
             f"marked include ({n_include_rows}); update the counts if this is intentional curation", "low")
    return out
