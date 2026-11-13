"""Smoke test of the WHOLE graph (planner -> ... -> report) with fake LLM and database replies."""
import json
import re
from pathlib import Path

import pytest

import src.agents.critic as crit_mod
import src.agents.extraction as ex_mod
import src.agents.planner as planner_mod
import src.agents.retrieval as ret_mod
import src.agents.screening as scr_mod
import src.agents.search_strategy as ss_mod
import src.agents.synthesis as syn_mod
import test_synthesis_stack as T
from src.agents.extraction import ROB_ITEMS

ROOT = Path(__file__).resolve().parents[1]


def rec(i, src, title, year=2022):
    return {"id": f"{'pmid:' if src == 'pubmed' else 'W'}{i}", "source": src, "doi": f"10.1/{src}{i}", "title": title,
            "abstract": f"Abstract of {title}. The model reached accuracy 9{i % 9}.0% on BUSI with 780 images.",
            "year": year, "authors": ["Ann Smith", "Bob Lee"], "venue": "J Test"}


PUBMED = [rec(i, "pubmed", f"Deep CNN for breast ultrasound classification {i}") for i in range(1, 7)] + \
         [rec(7, "pubmed", "A review of deep learning in imaging")]
OPENALEX = [rec(i, "openalex", f"3D CNN for CT lung nodule classification {i}") for i in range(1, 6)]


def fake_chat_json(system, user, **k):
    if "methodologist" in system:
        return {"title": "t", "pico": {"population": "p"}, "study_design": [], "inclusion_criteria": ["deep learning on images"],
                "exclusion_criteria": ["review articles"]}
    if "search strategies" in system:
        return {"pubmed": "(a[tiab]) AND (b[tiab])", "openalex": "deep learning image classification"}
    if "TITLE/ABSTRACT screening" in system:
        out = []
        for m in re.finditer(r"^\[(\d+)\] Title: (.+)$", user, re.M):
            exc = "review" in m.group(2).lower()
            out.append({"id": m.group(1), "decision": "exclude" if exc else "include",
                        "reason_code": "E1" if exc else "none", "rationale": "ok"})
        return {"decisions": out}
    if "senior arbiter" in system:
        return {"decision": "include", "reason_code": "none", "rationale": "ok"}
    if "data-extraction reviewer" in system:
        return {"eligibility": {"decision": "include", "reason_code": "none", "rationale": "ok"},
                "fields": {"dataset": {"value": "BUSI", "source": "abstract"},
                           "sample_size": {"value": "780 images", "source": "abstract"},
                           "method": {"value": "CNN", "source": "abstract"},
                           "metrics": {"value": "accuracy 90.0%", "source": "abstract"},
                           "main_findings": {"value": "good", "source": "abstract"},
                           "limitations": {"value": "single dataset", "source": "abstract"},
                           "study_design": {"value": "retrospective", "source": "abstract"}},
                "risk_of_bias": {k_: {"rating": "low", "note": ""} for k_ in ROB_ITEMS}}
    return T.fake_chat_json_syn(system, user, **k)          # grouping labels / assignments


def test_whole_graph_runs_from_question_to_report(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    from src.graph import build_graph
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    monkeypatch.delenv("HUMAN_APPROVAL", raising=False)
    for mod in (planner_mod, ss_mod, scr_mod, ex_mod, syn_mod, crit_mod):
        monkeypatch.setattr(mod, "chat_json", fake_chat_json)
    monkeypatch.setattr(syn_mod, "chat", T.make_chat())
    monkeypatch.setattr(ss_mod.pubmed, "count", lambda q, a, b: 500)
    monkeypatch.setattr(ret_mod.pubmed, "search", lambda q, n, a, b: PUBMED)
    monkeypatch.setattr(ret_mod.openalex, "search", lambda q, n, a, b: OPENALEX)
    monkeypatch.setattr(ex_mod, "fetch_all", lambda recs, d: {r["id"]: (None, {"reason": "test"}) for r in recs})
    run = tmp_path / "run_x"
    run.mkdir()
    final = build_graph().invoke({"run_id": "run_x", "run_dir": str(run), "question": "Deep learning for image classification?"})
    pr = final["prisma"]
    assert pr["total_identified"] == 12 and pr["records_after_dedup"] == 12
    assert pr["screened_title_abstract"] == 12 and pr["excluded_title_abstract"] == 1 and pr["included_for_fulltext"] == 11
    assert pr["included_in_review"] == 11 and pr["fulltext_retrieved"] == 0
    assert len(final["synthesis"]["studies"]) == 11 and {g["label"] for g in final["synthesis"]["groups"]} >= {"Ultrasound", "CT"}
    md = open(final["report_path"], encoding="utf-8").read()
    for needle in ("## 1. Summary of findings", "## 3. Results", "## 6. Quality assurance", "prisma_flow.svg",
                   "Citation coverage:", "(deep learning on images)" if False else "deep learning on images"):
        assert needle in md, needle
    timing = json.load(open(run / "timing.json"))
    assert set(timing) >= {"planner", "search_strategy", "retrieval", "screening", "extraction", "synthesis", "critic", "report"}
    for name in ("protocol.json", "queries.json", "records.json", "screening.json", "extraction.json", "evidence_table.csv",
                 "synthesis.json", "critic_report.json", "final_report.md", "references.bib", "decisions.jsonl"):
        assert (run / name).exists(), name
