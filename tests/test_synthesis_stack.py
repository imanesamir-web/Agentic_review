import json
import re

import pytest

import src.agents.critic as crit_mod
import src.agents.synthesis as syn_mod
from src.agents.extraction import ROB_ITEMS
from src.synthesis_utils import (parse_metrics, build_comparability, numbers_in, number_supported,
                                 numeric_values, split_sentences, cited_sids, eval_key, prisma_checks)

PROTO = {"research_question": "Deep learning for medical image classification?", "pico": {},
         "inclusion_criteria": ["a"], "exclusion_criteria": ["b"],
         "extraction_fields": ["study_design", "dataset", "sample_size", "method", "metrics", "main_findings",
                               "limitations"],
         "databases": ["pubmed", "openalex"], "date_range": {"from": 2015, "to": 2026},
         "max_records_per_database": 50}
PRISMA = {"identified": {"pubmed": 50, "openalex": 60}, "total_identified": 110, "duplicates_removed": 2,
          "records_after_dedup": 108, "screened_title_abstract": 108, "excluded_title_abstract": 100,
          "included_for_fulltext": 8, "fulltext_sought": 8, "fulltext_retrieved": 1,
          "assessed_on_abstract_only": 7, "excluded_after_fulltext_stage": 1, "included_in_review": 7,
          "cohen_kappa": 0.9, "exclusion_reasons": {"E1: b": 100}}


def row(i, title, year, dataset, n, method, metrics, lim, decision="include", failed=False, text="abstract_only"):
    f = lambda v: {"value": v, "source": "abstract"}
    items = {k: {"rating": "unclear" if k == "analysis_validation" else "low", "note": ""} for k in ROB_ITEMS}
    nr = "not reported"
    return {"id": i, "title": title, "year": year, "doi": f"10.1/{i}", "text_source": text,
            "extraction_failed": failed,
            "eligibility": {"decision": decision, "reason_code": "none", "reason": "", "rationale": ""},
            "fields": {"study_design": f("retrospective" if not failed else nr), "dataset": f(dataset),
                       "sample_size": f(n), "method": f(method), "metrics": f(metrics),
                       "main_findings": f("good performance" if not failed else nr), "limitations": f(lim)},
            "risk_of_bias": items, "overall_rob": "some concerns"}


ROWS = [
    row("a1", "Deep CNN for breast ultrasound classification", 2021, "BUSI", "780 images", "ResNet50",
        "accuracy 91.2%, AUC 0.95", "single dataset"),
    row("a2", "Transformer breast ultrasound", 2022, "BUSI", "780 images", "ViT", "Accuracy of 93.5%", "no external test"),
    row("a3", "Ensemble ultrasound", 2023, "BUSI", "780 images", "ensemble", "accuracy 89.0%", "small data"),
    row("a4", "Private ultrasound DL", 2023, "private hospital dataset", "500 images", "CNN", "accuracy 95.0%", "private data"),
    row("b1", "CT lung nodule classification", 2022, "LIDC-IDRI", "1018 scans", "3D CNN",
        "AUC 0.93; sensitivity 88%", "retrospective", text="full_text"),
    row("b2", "MRI brain tumor classifier", 2024, "BraTS", "285 cases", "CNN", "accuracy 97.1%", "small cohort"),
    row("x1", "Excluded study", 2020, "d", "1", "m", "accuracy 50%", "l", decision="exclude"),
    row("f1", "Failed extraction", 2020, "not reported", "not reported", "not reported", "not reported",
        "not reported", failed=True),
]
RECORDS = [{"id": r["id"], "title": r["title"], "year": r["year"], "doi": f"10.1/{r['id']}",
            "authors": ["Ann Smith", "Bob Lee"], "venue": "J Test"} for r in ROWS]


@pytest.fixture
def run(tmp_path, monkeypatch):
    d = tmp_path / "run"
    d.mkdir()
    json.dump(ROWS, open(d / "extraction.json", "w"))
    json.dump(RECORDS, open(d / "included_records.json", "w"))
    (d / "curation_note.txt").write_text("Extracted values were reviewed and corrected by the authors.")
    monkeypatch.setenv("CRITIC_SEMANTIC", "1")
    monkeypatch.setenv("MAX_REVISIONS", "2")
    return str(d)


# ---------- fakes ----------
def fake_chat_json_syn(system, user, **k):
    if "Propose 3 to 7" in system:
        return {"groups": ["Ultrasound", "CT", "MRI"]}
    out = []
    for m in re.finditer(r"^\[(\d+)\] (.+)$", user, re.M):
        line = m.group(2)
        g = "Ultrasound" if "ltrasound" in line else "CT" if "CT " in line else "MRI"
        ev = "BUSI" if "BUSI" in line else "LIDC-IDRI" if "LIDC" in line else "BraTS" if "BraTS" in line else "UNSPECIFIED"
        out.append({"id": m.group(1), "group": g, "eval_set": ev})
    return {"assignments": out}


def clean_group_text(user):
    sents = []
    for m in re.finditer(r"^\[(S\d+)\] .*?\| metrics: (.*?) \| findings", user, re.M):
        sents.append(f"Study {m.group(1)} reported {m.group(2)} [{m.group(1)}].")
    r = re.search(r"(\w+) on (\S+) \(n=(\d+)\): ([\d.]+)-([\d.]+)% \[([^\]]+)\]", user)
    if r:
        sents.append(f"{r.group(1)} ranged from {r.group(4)}% to {r.group(5)}% on {r.group(2)} [{r.group(6)}].")
    return " ".join(sents)


def make_chat(bad_group=None, always_bad=False):
    def fake_chat(system, user, **k):
        if "Results synthesis" in system:
            grp = re.search(r"^Group: (.+)$", user, re.M).group(1)
            bad = (always_bad or (bad_group == grp and "PREVIOUS DRAFT" not in user))
            text = clean_group_text(user)
            if bad:
                text += " The average accuracy was 92% overall. One study reached an accuracy of 99.9% [S01]."
            return text
        if "Summary of findings" in system:
            return "The evidence base is heterogeneous [S01, S02]. No studies were pooled."
        return "Single-dataset evaluation and small samples were common limitations [S01, S04]."
    return fake_chat


def fake_critic_json(system, user, **k):
    return {"issues": [], "verdict": "pass"}


def patch(monkeypatch, chat):
    monkeypatch.setattr(syn_mod, "chat_json", fake_chat_json_syn)
    monkeypatch.setattr(syn_mod, "chat", chat)
    monkeypatch.setattr(crit_mod, "chat_json", fake_critic_json)


def state_for(run):
    return {"run_id": "r", "run_dir": run, "protocol": PROTO, "prisma": dict(PRISMA), "queries": {"pubmed": "q1", "openalex": "q2"},
            "revision_count": 0}


# ---------- unit tests ----------
def test_parse_metrics():
    m, amb = parse_metrics("accuracy 91.2%, AUC 0.95 and specificity of 88%")
    assert m == {"Accuracy": 91.2, "AUC": 95.0, "Specificity": 88.0} and amb == []
    m, amb = parse_metrics("94.2% accuracy and 0.97 AUC")
    assert m == {"Accuracy": 94.2, "AUC": 97.0}
    m, amb = parse_metrics("accuracy 90% on A and accuracy 85% on B")
    assert m == {} and amb == ["Accuracy"]
    assert parse_metrics("not reported") == ({}, [])


def test_comparability_rules():
    mk = lambda sid, g, ev, acc: {"sid": sid, "group": g, "eval_set": ev, "eval_key": eval_key(ev),
                                  "metrics": {"Accuracy": acc}}
    sts = [mk("S01", "US", "BUSI", 90), mk("S02", "US", "busi", 92), mk("S03", "US", "BUSI", 88),
           mk("S04", "US", "private hospital data", 95), mk("S05", "CT", "LIDC", 90), mk("S06", "CT", "LIDC", 91)]
    comp = {(c["group"], c["metric"]): c for c in build_comparability(sts)}
    us, ct = comp[("US", "Accuracy")], comp[("CT", "Accuracy")]
    assert us["range_allowed"] and (us["min"], us["max"], us["n_cluster"]) == (88, 92, 3)
    assert not ct["range_allowed"] and "fewer than 3" in ct["reason"]


def test_number_and_sentence_helpers():
    allowed = numeric_values("accuracy 94.2% on 780 images")
    ok = lambda s: all(number_supported(v, d, p, allowed) for v, d, p, _ in numbers_in(s))
    assert ok("accuracy of 94% on 780 images") and ok("94.2%") and not ok("accuracy of 99.9%")
    assert ok("published in 2021") and ok("three of 7 studies")
    assert len(split_sentences("First one [S1]. Second 94.2% here [S2, S3]. Third.")) == 3
    assert cited_sids("x [S1, S12] y [S3]") == ["S1", "S12", "S3"]


def test_prisma_checks_detect_arithmetic_errors():
    bad = dict(PRISMA, duplicates_removed=5)
    assert any("duplicates" in d for _, _, d in prisma_checks(bad, 7))
    assert prisma_checks(PRISMA, 7) == []


# ---------- stage tests ----------
def test_synthesis_groups_tables_and_reuse(run, monkeypatch):
    patch(monkeypatch, make_chat())
    out = syn_mod.synthesis(state_for(run))["synthesis"]
    assert len(out["studies"]) == 6                       # excluded + failed rows are not synthesised
    assert {g["label"] for g in out["groups"]} == {"Ultrasound", "CT", "MRI"}
    us = next(g for g in out["groups"] if g["label"] == "Ultrasound")
    assert us["n"] == 4 and all(sid in out["tables"]["groups"]["Ultrasound"] for sid in us["sids"])
    acc = next(c for c in out["comparability"] if c["group"] == "Ultrasound" and c["metric"] == "Accuracy")
    assert acc["range_allowed"] and (acc["min"], acc["max"]) == (89.0, 93.5)
    assert "89.0–93.5% on BUSI" in out["tables"]["comparability"]
    assert out["studies"][0]["first_author"] == "Smith"
    # manual correction in group_assignments.json is honoured on the next run
    path = f"{run}/group_assignments.json"
    ga = json.load(open(path))
    ga["assignments"]["b2"]["group"] = "CT"
    json.dump(ga, open(path, "w"))
    out2 = syn_mod.synthesis(state_for(run))["synthesis"]
    assert next(s for s in out2["studies"] if s["id"] == "b2")["group"] == "CT"


def test_critic_flags_bad_text_and_passes_clean_text(run, monkeypatch):
    patch(monkeypatch, make_chat(bad_group="Ultrasound"))
    st = state_for(run)
    st.update(syn_mod.synthesis(st))
    res = crit_mod.critic(st)
    kinds = {(i["section"][:2], i["kind"]) for i in res["critic"]["issues"] if i["severity"] == "high"}
    assert ("G1", "number_unsupported") in kinds and ("G1", "pooling_language") in kinds
    assert res["needs_revision"] and res["critic_feedback"] and res["revision_count"] == 1
    patch(monkeypatch, make_chat())
    st2 = state_for(run)
    st2.update(syn_mod.synthesis(st2))
    clean = crit_mod.critic(st2)
    assert not clean["needs_revision"] and clean["critic"]["counts"]["high"] == 0


def test_critic_semantic_issue_is_matched_to_sentence(run, monkeypatch):
    monkeypatch.setenv("CRITIC_LLM_REWRITES", "1")
    patch(monkeypatch, make_chat())
    st = state_for(run)
    st.update(syn_mod.synthesis(st))
    first = next(b for b in st["synthesis"]["blocks"] if b["kind"] == "group")
    target = first["text"].split(". ")[0].rstrip(".") + "."
    monkeypatch.setattr(crit_mod, "chat_json", lambda s, u, **k: (
        {"issues": [{"sentence": target, "problem": "overstated", "severity": "high"}], "verdict": "revise"}
        if first["key"] in u.split("TEXT:")[0] else {"issues": [], "verdict": "pass"}))
    res = crit_mod.critic(st)
    sem = [i for i in res["critic"]["issues"] if i["source"] == "llm" and i["kind"] == "unsupported_claim"]
    assert sem and sem[0]["sentence"] == target and res["needs_revision"]


def test_graph_revision_loop_fixes_text_and_writes_report(run, monkeypatch):
    pytest.importorskip("langgraph")
    from src.graph import build_synthesis_graph
    patch(monkeypatch, make_chat(bad_group="Ultrasound"))
    final = build_synthesis_graph().invoke(state_for(run))
    assert final["revision_count"] == 1 and final["critic"]["verdict"] == "pass"
    md = open(final["report_path"], encoding="utf-8").read()
    assert "⚠" not in md and "99.9" not in md
    assert "## 6. Quality assurance" in md and "89.0–93.5% on BUSI" in md and "reviewed and corrected" in md
    assert "pass 1" in md and "pass 2" in md
    for name in ("prisma_flow.svg", "references.bib", "critic_report.json", "synthesis_table.csv",
                 "final_report.html", "group_assignments.json"):
        assert (__import__("pathlib").Path(run) / name).exists(), name
    svg = open(f"{run}/prisma_flow.svg", encoding="utf-8").read()
    assert "Records identified: 110" in svg and "Studies synthesised: 6" in svg
    assert "@article{S01," in open(f"{run}/references.bib", encoding="utf-8").read()


def test_unresolved_issues_are_disclosed_not_hidden(run, monkeypatch):
    pytest.importorskip("langgraph")
    from src.graph import build_synthesis_graph
    monkeypatch.setenv("MAX_REVISIONS", "1")
    patch(monkeypatch, make_chat(always_bad=True))
    final = build_synthesis_graph().invoke(state_for(run))
    assert final["critic"]["verdict"] == "pass_with_unresolved_issues"
    md = open(final["report_path"], encoding="utf-8").read()
    assert "⚠[unverified]" in md and "pass_with_unresolved_issues" in md
