import json
import re

import pytest

import src.agents.critic as crit_mod
import src.agents.synthesis as syn_mod
import test_synthesis_stack as T


def build_state(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, T.make_chat())
    st = T.state_for(str(run))
    st.update(syn_mod.synthesis(st))
    return st


def set_text(st, text):
    block = next(b for b in st["synthesis"]["blocks"] if b["kind"] == "group")
    block["text"] = text
    return block


def kinds(res, key):
    return {i["kind"] for i in res["critic"]["issues"] if i["section"] == key and i["severity"] == "high"}


def test_inherit_scope_accepts_uncited_sentences_but_checks_cited_ones_strictly(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    monkeypatch.setenv("CRITIC_CITATION_SCOPE", "inherit")
    st = build_state(tmp_path, monkeypatch)
    us = next(s["sid"] for s in st["synthesis"]["studies"] if "Deep CNN" in s["title"])
    ct = next(s["sid"] for s in st["synthesis"]["studies"] if "CT lung" in s["title"])
    # 1) faithful sentence whose citation is only at the end of the paragraph: fine under 'inherit'
    block = set_text(st, f"Study {us} reported accuracy 91.2%, AUC 0.95. It was evaluated on BUSI. See details [{us}].")
    assert "number_unsupported" not in kinds(crit_mod.critic(st), block["key"])
    # 2) cited sentence using a number of ANOTHER study cited in the paragraph: caught under 'inherit'
    block = set_text(st, f"Study {us} analysed 1018 scans [{us}]. A different cohort was used [{ct}].")
    assert "number_unsupported" in kinds(crit_mod.critic(st), block["key"])


def test_llm_findings_can_be_advisory(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SEMANTIC", "1")
    st = build_state(tmp_path, monkeypatch)
    monkeypatch.setattr(crit_mod, "chat_json", lambda s, u, **k: {
        "issues": [{"sentence": "x", "problem": "overstated", "severity": "high"}], "verdict": "revise"})
    monkeypatch.setenv("CRITIC_LLM_REWRITES", "1")
    res = crit_mod.critic(st)
    assert res["needs_revision"] and res["critic"]["config"]["llm_findings_trigger_rewrites"] is True
    st["revision_count"] = 0
    monkeypatch.setenv("CRITIC_LLM_REWRITES", "0")
    res = crit_mod.critic(st)
    assert not res["needs_revision"] and res["critic"]["config"]["llm_findings_trigger_rewrites"] is False
    assert res["critic"]["counts"]["high"] > 0 and res["critic"]["verdict"] == "pass_with_unresolved_issues"


@pytest.mark.parametrize("scope,phrase", [("inherit", "a sentence without a citation of its own"),
                                          ("sentence", "against the studies cited in the same sentence")])
def test_report_describes_scope_and_advisory_llm(tmp_path, monkeypatch, scope, phrase):
    pytest.importorskip("langgraph")
    from src.graph import build_synthesis_graph
    monkeypatch.setenv("CRITIC_CITATION_SCOPE", scope)
    monkeypatch.setenv("CRITIC_SEMANTIC", "1")
    monkeypatch.setenv("CRITIC_LLM_REWRITES", "0")
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, T.make_chat())
    final = build_synthesis_graph().invoke(T.state_for(str(run)))
    md = open(final["report_path"], encoding="utf-8").read()
    assert phrase in md and "did not trigger rewrites" in md


def test_writer_prompt_forbids_generalising_about_the_group():
    assert "Do NOT generalise about the whole group" in syn_mod.SYS_GROUP
