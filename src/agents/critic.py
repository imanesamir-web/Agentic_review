"""Critic / QA agent. Checks that the synthesis matches the extracted evidence and the protocol:
Deterministic checks cover citations, numbers, pooling language, PRISMA arithmetic, coverage and scope.
Settings (all recorded in critic_report.json and described in the report):
  CRITIC_SEMANTIC=1         also run an LLM check of every passage against the studies it cites (default 0)
  CRITIC_CITATION_SCOPE     'paragraph' (default): a sentence is checked against the studies cited anywhere in its
                            paragraph; 'inherit': a cited sentence is checked against its own citations, an uncited
                            sentence against its paragraph's; 'sentence': only its own citations (strictest)
  CRITIC_LLM_REWRITES=0     LLM findings are listed in the report but do not trigger rewrites (default 1)
  CRITIC_MODEL              model for the LLM check (default: the main model)
  CRITIC_EVIDENCE_CHARS     characters per extracted field shown to the LLM critic (default 450)
  MAX_REVISIONS             rewrite passes requested for high-severity findings (default 2)
Findings that remain are disclosed in the report."""
import copy
import difflib
import os

from src.llm import chat_json
from src.audit import log, save_json
from src.synthesis_utils import (cited_sids, strip_cites, split_sentences, numbers_in, numeric_values,
                                 number_supported, study_numbers, POOL_RE, NEGATION_RE, OVERCLAIM_RE,
                                 compact, prisma_checks, rob_inconsistent)

SYS_CRITIC = (
    "You are the quality-assurance critic of a systematic review. Compare the TEXT with the EVIDENCE (extracted "
    "records of the studies it cites). Report only REAL problems: (1) a claim not supported by the cited studies, "
    "(2) a wrong study cited, (3) a statement overstated or generalised beyond the evidence, (4) numbers compared, "
    "averaged or ranked across studies without justification, (5) a claim outside the review scope. "
    'Return {"issues":[{"sentence":"<copy the problematic sentence exactly>","problem":"...","severity":"high|medium|low"}],'
    '"verdict":"pass|revise"}. If the text is faithful return {"issues":[],"verdict":"pass"}.')


def _issue(section, kind, severity, detail, sentence=None, source="deterministic"):
    return {"section": section, "kind": kind, "severity": severity, "detail": detail,
            "sentence": sentence, "source": source}


def citation_scope():
    v = os.getenv("CRITIC_CITATION_SCOPE", "paragraph").strip().lower()
    return v if v in ("paragraph", "inherit", "sentence") else "paragraph"


def critic_config():
    return {"semantic_llm_check": os.getenv("CRITIC_SEMANTIC", "0") == "1",
            "critic_model": os.getenv("CRITIC_MODEL") or os.getenv("LLM_MODEL") or "n/a",
            "citation_scope": citation_scope(),
            "pooling_rule": "pooling/averaging wording is flagged when the sentence cites several studies or names "
                            "meta-analysis / pooled estimate / pooled analysis",
            "llm_findings_trigger_rewrites": os.getenv("CRITIC_LLM_REWRITES", "1") == "1",
            "max_revisions": int(os.getenv("MAX_REVISIONS", "2"))}


def check_block(blk, studies):
    key, text, out = blk["key"], blk["text"], []
    if blk.get("error"):
        out.append(_issue(key, "generation_failed", "high", f"The passage could not be generated: {blk['error']}"))
    allowed_group = set(blk["sids"]) if blk["kind"] == "group" else set(studies)
    facts_vals = numeric_values(blk.get("facts", ""))
    scope = citation_scope()
    for line in text.splitlines():
        line_cites = cited_sids(line)
        for s in split_sentences(line):
            own = list(dict.fromkeys(cited_sids(s)))      # the same study cited twice is still ONE study
            if scope == "paragraph":        # any study cited in the paragraph supports the sentence
                cites = list(dict.fromkeys(own + line_cites))
            elif scope == "inherit":        # a cited sentence is checked on its own citations; an uncited one inherits
                cites = own if own else list(dict.fromkeys(line_cites))
            else:                           # strict: only the sentence's own citations
                cites = own
            body = strip_cites(s)
            unknown = [c for c in cites if c not in studies]
            if unknown:
                out.append(_issue(key, "unknown_citation", "high", f"cites non-existent study ids {unknown}", s))
            outside = [c for c in cites if c in studies and c not in allowed_group]
            if outside:
                out.append(_issue(key, "wrong_group_citation", "medium", f"cites studies outside this group: {outside}", s))
            nums = numbers_in(body)
            bad = []
            if nums:
                allowed = set(facts_vals)
                for c in cites:
                    if c in studies:
                        allowed |= study_numbers(studies[c])
                bad = [raw for v, d, pct, raw in nums if not number_supported(v, d, pct, allowed)]
                if bad:
                    out.append(_issue(key, "number_unsupported", "high",
                                      f"numbers {bad} are not in the cited studies' extracted data or the FACTS", s))
            uncited = (not cites) if scope == "paragraph" else (not own)
            if blk["kind"] == "group" and uncited and len(body.split()) >= 8 and not (nums and not bad):
                out.append(_issue(key, "uncited_claim", "low", "factual sentence without a study citation", s))
            if (POOL_RE.search(body) and not NEGATION_RE.search(body) and
                    (len(cites) > 1 or any(term in body.lower() for term in ("meta-analysis", "pooled estimate", "pooled analysis")))):
                out.append(_issue(key, "pooling_language", "high",
                                  "pooling/averaging language: metrics must not be combined across studies", s))
            if OVERCLAIM_RE.search(body):
                out.append(_issue(key, "overclaim", "medium", "overstated wording", s))
    return out


def _match_sentence(text, claimed):
    if not claimed:
        return None
    sents = split_sentences(text)
    if claimed in sents:
        return claimed
    best = max(sents, key=lambda x: difflib.SequenceMatcher(None, x, claimed).ratio(), default=None)
    if best and difflib.SequenceMatcher(None, best, claimed).ratio() >= 0.8:
        return best
    return None


def semantic_check(blk, studies, protocol):
    cited = list(dict.fromkeys(c for c in cited_sids(blk["text"]) if c in studies)) or blk["sids"]
    chars = int(os.getenv("CRITIC_EVIDENCE_CHARS", "450"))    # per field; 200 cut real metrics and caused false alarms
    evidence = "\n".join(compact(studies[c], chars) for c in cited[:6])
    user = (f"Review question: {protocol['research_question']}\nSection: {blk['key']}\n\nTEXT:\n{blk['text']}\n\n"
            f"EVIDENCE:\n{evidence}")
    try:
        d = chat_json(SYS_CRITIC, user, model=os.getenv("CRITIC_MODEL") or None)
    except Exception as e:
        return [_issue(blk["key"], "semantic_check_failed", "low", f"LLM check unavailable: {str(e)[:120]}",
                       source="llm")]
    out = []
    for it in d.get("issues", []) or []:
        sev = str(it.get("severity", "medium")).lower()
        out.append(_issue(blk["key"], "unsupported_claim",
                          sev if sev in ("high", "medium", "low") else "medium",
                          str(it.get("problem", ""))[:300], _match_sentence(blk["text"], it.get("sentence")), "llm"))
    return out


def global_checks(syn, protocol, prisma, rows_curated):
    out, studies = [], syn["studies"]
    for kind, sev, detail in prisma_checks(prisma, syn["n_include_rows"]):
        out.append(_issue("global", kind, sev, detail))
    sids = [x for g in syn["groups"] for x in g["sids"]]
    if sorted(sids) != sorted(s["sid"] for s in studies):
        out.append(_issue("global", "group_coverage", "high", "every study must belong to exactly one group"))
    for g in syn["groups"]:
        if not any(b["kind"] == "group" and b["group"] == g["label"] and b["text"].strip() for b in syn["blocks"]):
            out.append(_issue("global", "group_without_narrative", "high", f"group '{g['label']}' has no narrative"))
    kinds = {b["kind"] for b in syn["blocks"] if b["text"].strip()}
    for need in ("summary", "limitations"):
        if need not in kinds:
            out.append(_issue("global", "missing_section", "high", f"required section '{need}' is empty"))
    dr = protocol.get("date_range") or {}
    for s in studies:
        if s["year"] and dr and not (dr.get("from", 0) <= s["year"] <= dr.get("to", 9999)):
            out.append(_issue("global", "outside_date_range", "medium",
                              f"{s['sid']} ({s['year']}) is outside the protocol date range {dr}"))
    for r in rows_curated:
        inc = rob_inconsistent(r)
        if inc:
            out.append(_issue("global", "rob_inconsistent", "low",
                              f"'{r.get('title', '')[:60]}': stored overall RoB '{inc[0]}' differs from its item "
                              f"ratings ('{inc[1]}'); check the curated rating"))
    for c in syn["comparability"]:
        if c["range_allowed"] and (c["n_cluster"] < 3 or not c.get("eval_set")):
            out.append(_issue("global", "comparability_rule", "high",
                              f"range for {c['metric']} in {c['group']} violates the comparability rule"))
    unc = sorted({s["sid"] for s in studies} - {c for b in syn["blocks"] for c in cited_sids(b["text"])})
    if unc:
        out.append(_issue("global", "uncited_studies", "low", f"studies never cited in the narrative: {unc}"))
    return out


def critic(state):
    p, run_dir, syn = state["protocol"], state["run_dir"], state["synthesis"]
    studies = {s["sid"]: s for s in syn["studies"]}
    issues = []
    for blk in syn["blocks"]:
        issues += check_block(blk, studies)
        if os.getenv("CRITIC_SEMANTIC", "0") == "1":
            issues += semantic_check(blk, studies, p)
    from src.agents.synthesis import load_rows
    issues += global_checks(syn, p, state.get("prisma", {}), load_rows(state))
    block_keys = {b["key"] for b in syn["blocks"]}
    llm_rewrites = os.getenv("CRITIC_LLM_REWRITES", "1") == "1"

    def triggers(i):    # which findings may request a rewrite (LLM findings can be made advisory)
        return i["severity"] == "high" and i["section"] in block_keys and (llm_rewrites or i["source"] != "llm")

    high = [i for i in issues if triggers(i)]
    sev = {k: sum(1 for i in issues if i["severity"] == k) for k in ("high", "medium", "low")}
    rev = state.get("revision_count", 0)
    max_rev = int(os.getenv("MAX_REVISIONS", "2"))
    best = state.get("best_pass")
    # a rewrite is only kept if it has FEWER high-severity findings than the best version so far
    rejected = best is not None and len(high) >= best["n_high"]
    if rejected:
        chosen = best
    else:
        chosen = {"n_high": len(high), "issues": issues, "sev": sev, "synthesis": copy.deepcopy(syn), "pass": rev + 1}
    chosen_high = [i for i in chosen["issues"] if triggers(i)]
    any_high = [i for i in chosen["issues"] if i["severity"] == "high" and i["section"] in block_keys]
    needs = (not rejected) and bool(chosen_high) and rev < max_rev
    feedback = {}
    if needs:
        for i in chosen_high:
            feedback.setdefault(i["section"], []).append(
                f"{i['kind']}: {i['detail']}" + (f" -- sentence: \"{i['sentence']}\"" if i["sentence"] else ""))
    entry = {"pass": rev + 1, "issues": sev}
    if rejected:
        entry["note"] = f"rewrite rejected: not better than pass {best['pass']}; best version kept"
    history = list((state.get("critic") or {}).get("history", [])) + [entry]
    report = {"issues": chosen["issues"], "counts": chosen["sev"], "history": history,
              "verdict": "revise" if needs else ("pass" if not any_high else "pass_with_unresolved_issues"),
              "max_revisions": max_rev, "config": critic_config(), "best_pass": chosen["pass"]}
    out_syn = chosen["synthesis"] if rejected else syn
    if rejected:
        save_json(run_dir, "synthesis.json", out_syn)       # the rejected rewrite must not stay on disk
    save_json(run_dir, "critic_report.json", report)
    log(run_dir, "critic", "pass", {"counts": sev, "needs_revision": needs, "rewrite_rejected": rejected})
    print(f"[critic] pass {rev + 1}: {sev} -> {report['verdict']}" +
          (f"  (rewrite rejected, best version of pass {best['pass']} kept)" if rejected else ""), flush=True)
    return {"critic": report, "needs_revision": needs, "critic_feedback": feedback if needs else {},
            "revision_count": rev + (1 if needs else 0), "best_pass": chosen, "synthesis": out_syn}
