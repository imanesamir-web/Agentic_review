"""Report stage: final Markdown (and HTML) report, BibTeX, PRISMA flow (SVG + JSON) and the QA appendix.
Everything here is deterministic: it only assembles what the earlier stages produced."""
import json
import os
import re
from collections import Counter
from xml.sax.saxutils import escape

from src.audit import log, save_json
from src.fileutil import safe_path
from src.synthesis_utils import rob_counts, ROB_ITEMS, cited_sids, split_sentences

ITEM_HELP = {
    "selection_sampling": "representative, clearly described data/participants",
    "method_description": "method is described well enough to reproduce",
    "outcome_measurement": "appropriate reference standard / outcome",
    "analysis_validation": "proper train/test split, no leakage, external validation",
    "reporting_completeness": "completeness of reporting",
}


def _n(p, k):
    v = p.get(k)
    return "n/a" if v is None else v


def prisma_svg(p, n_synth):
    ident = p.get("identified", {})
    boxes = [
        ("Identification", [f"Records identified: {_n(p, 'total_identified')}",
                            " | ".join(f"{k} {v}" for k, v in ident.items()) or "databases: n/a",
                            f"Duplicates removed: {_n(p, 'duplicates_removed')}"]),
        ("Screening (title/abstract)", [f"Records screened: {_n(p, 'screened_title_abstract')}",
                                        f"Excluded: {_n(p, 'excluded_title_abstract')}"]),
        ("Eligibility (full-text stage)", [f"Reports sought: {_n(p, 'fulltext_sought')}",
                                           f"Full text retrieved: {_n(p, 'fulltext_retrieved')}",
                                           f"Assessed on abstract only: {_n(p, 'assessed_on_abstract_only')}",
                                           f"Excluded: {_n(p, 'excluded_after_fulltext_stage')}"]),
        ("Included", [f"Studies included in review: {_n(p, 'included_in_review')}",
                      f"Studies synthesised: {n_synth}"]),
    ]
    w, y, parts = 520, 10, []
    for title, lines in boxes:
        h = 34 + 20 * len(lines)
        parts.append(f'<rect x="20" y="{y}" width="{w}" height="{h}" rx="6" fill="#f6f6f6" stroke="#444"/>')
        parts.append(f'<text x="34" y="{y + 22}" font-family="sans-serif" font-size="14" font-weight="bold" '
                     f'fill="#111">{escape(title)}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="34" y="{y + 44 + 20 * i}" font-family="sans-serif" font-size="13" '
                         f'fill="#111">{escape(str(line))}</text>')
        y += h
        if title != "Included":
            parts.append(f'<line x1="280" y1="{y}" x2="280" y2="{y + 22}" stroke="#444"/>'
                         f'<polygon points="274,{y + 16} 286,{y + 16} 280,{y + 24}" fill="#444"/>')
            y += 24
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 {y + 10}" width="560" height="{y + 10}">'
            + "".join(parts) + "</svg>")


def _bib_escape(s):
    return str(s or "").replace("{", "(").replace("}", ")").replace("&", "\\&")


def bibtex(studies):
    out = []
    for s in studies:
        e = [f"@article{{{s['sid']},", f"  title = {{{_bib_escape(s['title'])}}},"]
        if s["authors"]:
            e.append(f"  author = {{{' and '.join(_bib_escape(a) for a in s['authors'])}}},")
        if s["year"]:
            e.append(f"  year = {{{s['year']}}},")
        if s["venue"]:
            e.append(f"  journal = {{{_bib_escape(s['venue'])}}},")
        if s["doi"]:
            e.append(f"  doi = {{{s['doi']}}},")
        out.append("\n".join(e) + "\n}")
    return "\n\n".join(out) + "\n"


def ref_line(s):
    who = (", ".join(s["authors"][:3]) + (" et al." if len(s["authors"]) > 3 else "")) if s["authors"] else "Authors n/a"
    doi = f" https://doi.org/{s['doi']}" if s["doi"] else ""
    return f"**[{s['sid']}]** {who} ({s['year']}). {s['title']}. {s['venue'] or ''}{doi}".replace(" .", ".")


def models_by_stage(run_dir):
    path = os.path.join(run_dir, "decisions.jsonl")
    c = Counter()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    c[(r.get("stage"), r.get("model") or "n/a")] += 1
                except Exception:
                    continue
    return c


def mark_unresolved(text, key, issues):
    for i in issues:
        if i["section"] == key and i["severity"] == "high" and i.get("sentence") and i["sentence"] in text:
            text = text.replace(i["sentence"], i["sentence"] + " ⚠[unverified]", 1)
    return text


def _list(items, prefix):
    return "\n".join(f"- **{prefix}{i + 1}**: {x}" for i, x in enumerate(items)) or "- (none)"


def critic_description(crit):
    """Plain-language description of what the critic really did (from its recorded configuration)."""
    cfg = crit.get("config")
    if not cfg:
        return "A critic agent checked the passages; its configuration was not recorded for this run."
    scope = {"paragraph": "against the studies cited anywhere in the same paragraph",
             "inherit": "against the studies cited in the same sentence (a sentence without a citation of its own is "
                        "checked against the studies cited in its paragraph)",
             "sentence": "against the studies cited in the same sentence"}[cfg["citation_scope"]]
    text = ("A critic agent ran deterministic checks: every citation must exist (and, within a group passage, belong "
            f"to that group); every number is looked for in the extracted data of the cited studies, {scope}, or in the "
            "listed facts; pooling or averaging wording is flagged when a sentence cites several studies or names "
            "meta-analysis; the PRISMA counts must add up; every study must be in exactly one group; studies must lie "
            "inside the protocol date range.")
    if cfg["semantic_llm_check"]:
        text += (f" Each passage was also compared with the studies it cites by an LLM critic ({cfg['critic_model']}).")
        if not cfg.get("llm_findings_trigger_rewrites", True):
            text += " The LLM critic's findings are listed in section 6 but did not trigger rewrites."
    else:
        text += (" No LLM-based semantic check was run: a sentence whose numbers are correct but whose wording "
                 "overstates or misreads a study is not detected automatically, so the narrative should be read "
                 "against the evidence tables.")
    return text + f" High-severity findings triggered up to {cfg['max_revisions']} rewrite passes."


def citation_coverage(syn):
    """(sentences with a citation of their own, all sentences) in the group passages."""
    total = cited = 0
    for b in syn["blocks"]:
        if b["kind"] != "group":
            continue
        for line in b["text"].splitlines():
            for sent in split_sentences(line):
                total += 1
                cited += bool(cited_sids(sent))
    return cited, total


def build_markdown(state, syn, crit, queries, curation_note):
    p, prisma = state["protocol"], state.get("prisma", {})
    studies, n = syn["studies"], len(syn["studies"])
    issues = crit["issues"]
    blocks = {b["key"]: b for b in syn["blocks"]}
    txt = lambda k: mark_unresolved(blocks[k]["text"], k, issues) if k in blocks else ""
    full = sum(1 for s in studies if s["text_source"] == "full_text")
    title = p.get("title") or f"Systematic review: {p['research_question']}"
    note = (" Passages marked ⚠[unverified] did not pass the automated quality check."
            if crit["verdict"] == "pass_with_unresolved_issues" else "")
    L = [f"# {title}", "", f"*Generated automatically by the agentic review pipeline (run `{state.get('run_id', '')}`; "
         f"PRISMA 2020-aligned).{note}*", ""]
    L += ["## 1. Summary of findings", "", txt("SUMMARY"), ""]
    L += ["## 2. Methods", "", f"**Research question.** {p['research_question']}", ""]
    pico = p.get("pico") or {}
    if pico:
        L += ["**PICO.** " + "; ".join(f"{k}: {v}" for k, v in pico.items() if v), ""]
    L += ["**Inclusion criteria**", _list(p.get("inclusion_criteria", []), "I"), "",
          "**Exclusion criteria**", _list(p.get("exclusion_criteria", []), "E"), "",
          f"**Sources and limits.** Databases: {', '.join(p.get('databases', []))}; publication years "
          f"{(p.get('date_range') or {}).get('from', 'n/a')}-{(p.get('date_range') or {}).get('to', 'n/a')}; "
          f"at most {p.get('max_records_per_database', 'n/a')} records retrieved per database.", ""]
    if queries:
        L += ["**Search strategies**", "", f"- PubMed: `{queries.get('pubmed', '')}`",
              f"- OpenAlex: `{queries.get('openalex', '')}`", ""]
    L += ["**Screening.** Two independent LLM reviewers screened titles and abstracts against the numbered criteria; "
          "disagreements were resolved by an arbiter agent; 'maybe' decisions went forward to the full-text stage. "
          f"Inter-reviewer agreement (Cohen's kappa): {_n(prisma, 'cohen_kappa')}. Kappa measures consistency, not accuracy.", "",
          "**Data extraction and appraisal.** An extraction agent re-checked eligibility and extracted "
          f"{', '.join(p.get('extraction_fields', []))} using open-access full text when available "
          f"({full} of {n} synthesised studies) and the abstract otherwise, then rated risk of bias on five items:", "",
          "\n".join(f"- *{k.replace('_', ' ')}*: {v}" for k, v in ITEM_HELP.items()), ""]
    if curation_note:
        L += [f"**Data curation.** {curation_note}", ""]
    L += ["**Synthesis.** Narrative synthesis, grouped by "
          f"{os.getenv('SYNTH_GROUP_BY', 'auto')}. Performance metrics were **not pooled**. A descriptive range is shown "
          "only when at least three studies report the same metric on the same named evaluation dataset; otherwise "
          "results are reported study by study. " + critic_description(crit) + " (Results: section 6.)", ""]
    L += ["## 3. Results", "", "### 3.1 Study selection", "",
          f"{_n(prisma, 'total_identified')} records were identified ("
          + ", ".join(f"{k}: {v}" for k, v in (prisma.get('identified') or {}).items()) +
          f"); {_n(prisma, 'duplicates_removed')} duplicates were removed and {_n(prisma, 'screened_title_abstract')} "
          f"records were screened. {_n(prisma, 'excluded_title_abstract')} were excluded at title/abstract level and "
          f"{_n(prisma, 'fulltext_sought')} proceeded to the full-text stage ({_n(prisma, 'fulltext_retrieved')} full texts "
          f"retrieved, {_n(prisma, 'assessed_on_abstract_only')} assessed on the abstract only). "
          f"{_n(prisma, 'excluded_after_fulltext_stage')} were excluded at that stage. "
          f"{_n(prisma, 'included_in_review')} studies were included, of which {n} are synthesised below.", "",
          "![PRISMA flow](prisma_flow.svg)", ""]
    reasons = {**{f"[title/abstract] {k}": v for k, v in (prisma.get('exclusion_reasons') or {}).items()},
               **{f"[full-text stage] {k}": v for k, v in (prisma.get('exclusion_reasons_fulltext_stage') or {}).items()}}
    if reasons:
        L += ["**Exclusion reasons**", "", "\n".join(f"- {k}: {v}" for k, v in reasons.items()), ""]
    L += ["### 3.2 Study characteristics and narrative synthesis", ""]
    for g in syn["groups"]:
        L += [f"#### {g['label']} ({g['n']} studies)", "", syn["tables"]["groups"][g["label"]], ""]
        for b in syn["blocks"]:
            if b["kind"] == "group" and b["group"] == g["label"]:
                L += [txt(b["key"]), ""]
    L += ["### 3.3 Comparability of reported metrics", "",
          "Metrics are reported as extracted from each study. No pooled estimate is given.", "",
          syn["tables"]["comparability"], "", "### 3.4 Risk of bias", "",
          "Counts per group; item columns show studies rated low / unclear / high.", "", syn["tables"]["rob"], ""]
    abstract_only = n - full
    L += ["## 4. Limitations", "", txt("LIMITATIONS"), "", "**Review-level limitations**", "",
          f"- The search used {len(p.get('databases', []))} databases and at most {p.get('max_records_per_database', 'n/a')} "
          "records per database, so the review is not exhaustive.",
          f"- {abstract_only} of {n} synthesised studies were assessed on the abstract only; for these, risk-of-bias "
          "ratings mostly reflect what the abstract reports, not how the study was conducted.",
          "- Screening, extraction and appraisal were performed by LLM agents; agreement between agents is not a "
          "measure of correctness without a human reference standard.",
          "- Metrics differ in definition, datasets and evaluation protocol across studies, so no quantitative "
          "synthesis was attempted.", ""]
    L += ["## 5. References", "", "\n\n".join(ref_line(s) for s in studies), ""]
    c = crit["counts"]
    unresolved = [i for i in issues if i["severity"] in ("high", "medium")]
    L += ["## 6. Quality assurance", "",
          f"The critic agent made {len(crit['history'])} pass(es); final verdict: **{crit['verdict']}**. "
          f"Remaining findings: {c['high']} high, {c['medium']} medium, {c['low']} low severity. "
          "History: " + "; ".join(f"pass {h['pass']}: {h['issues']}" + (f" ({h['note']})" if h.get("note") else "")
                                  for h in crit["history"]) + ".", "",
          "Citation coverage: {} of {} sentences in the group passages carry a citation of their own ({}%); "
          "the others rely on the citations of their paragraph.".format(
              *citation_coverage(syn), round(100 * citation_coverage(syn)[0] / max(citation_coverage(syn)[1], 1))), ""]
    if unresolved:
        L += ["| Section | Check | Severity | Detail |", "|---|---|---|---|"]
        for i in unresolved[:40]:
            L.append(f"| {i['section']} | {i['kind']} | {i['severity']} | "
                     f"{' '.join(str(i['detail']).split())[:160].replace('|', '/')} |")
        L.append("")
    L += ["## 7. Reproducibility", "",
          "Models recorded in the audit trail (`decisions.jsonl`; the logged name is the configured primary model, "
          "a fallback model may have answered when it was rate-limited):", ""]
    ms = models_by_stage(state["run_dir"])
    L += [f"- {st}: {m} ({k} decisions)" for (st, m), k in sorted(ms.items(), key=lambda x: str(x[0]))] or ["- n/a"]
    L += ["", f"Configured now: LLM_MODEL={os.getenv('LLM_MODEL', 'n/a')}, "
          f"SCREEN_MODEL_B={os.getenv('SCREEN_MODEL_B') or 'same'}, CRITIC_MODEL={os.getenv('CRITIC_MODEL') or 'same'}. "
          "Protocol, queries, records, screening decisions, extraction, grouping and the critic report are saved in the "
          "run folder.", ""]
    return "\n".join(L)


def report(state):
    run_dir, syn, crit = state["run_dir"], state["synthesis"], state["critic"]
    qpath = os.path.join(run_dir, "queries.json")
    queries = state.get("queries") or (json.load(open(qpath, encoding="utf-8")) if os.path.exists(qpath) else {})
    note_path = os.path.join(run_dir, "curation_note.txt")
    note = open(note_path, encoding="utf-8").read().strip() if os.path.exists(note_path) else ""
    md = build_markdown(state, syn, crit, queries, note)
    md_path = safe_path(os.path.join(run_dir, "final_report.md"))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    svg = prisma_svg(state.get("prisma", {}), len(syn["studies"]))
    with open(safe_path(os.path.join(run_dir, "prisma_flow.svg")), "w", encoding="utf-8") as f:
        f.write(svg)
    save_json(run_dir, "prisma_flow.json", {**state.get("prisma", {}), "synthesised": len(syn["studies"])})
    with open(safe_path(os.path.join(run_dir, "references.bib")), "w", encoding="utf-8") as f:
        f.write(bibtex(syn["studies"]))
    try:
        import markdown
        body = markdown.markdown(md, extensions=["tables"])
        body = body.replace('<img alt="PRISMA flow" src="prisma_flow.svg" />', svg)
        css = ("body{font-family:Georgia,serif;max-width:900px;margin:2em auto;line-height:1.5;padding:0 1em}"
               "table{border-collapse:collapse;font-size:12px;margin:1em 0}td,th{border:1px solid #999;padding:3px 6px}"
               "th{background:#eee}code{font-size:11px;word-break:break-all}")
        with open(safe_path(os.path.join(run_dir, "final_report.html")), "w", encoding="utf-8") as f:
            f.write(f"<!doctype html><meta charset='utf-8'><title>Systematic review</title><style>{css}</style>{body}")
    except ImportError:
        print("[report] 'markdown' package not installed: skipped final_report.html (pip install markdown)", flush=True)
    log(run_dir, "report", "written", {"studies": len(syn["studies"]), "verdict": crit["verdict"]})
    print(f"[report] wrote {md_path}", flush=True)
    return {"report_path": md_path}
