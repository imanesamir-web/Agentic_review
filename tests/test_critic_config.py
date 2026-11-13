import json

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


def attribution_text(st):
    """A paragraph where sentence 1 uses a number (1018 scans) that belongs to ANOTHER study (the CT study) cited in
    sentence 2 of the same paragraph."""
    ct = next(s["sid"] for s in st["synthesis"]["studies"] if "CT lung" in s["title"])
    us = next(s["sid"] for s in st["synthesis"]["studies"] if "Deep CNN" in s["title"])
    block = next(b for b in st["synthesis"]["blocks"] if b["kind"] == "group")
    block["text"] = f"Study {us} analysed 1018 scans [{us}]. A different study used a CT cohort [{ct}]."
    return block["key"]


def high_numbers(res, key):
    return [i for i in res["critic"]["issues"] if i["section"] == key and i["kind"] == "number_unsupported"]


def test_paragraph_scope_is_the_default_and_accepts_numbers_from_the_paragraph(tmp_path, monkeypatch):
    monkeypatch.delenv("CRITIC_CITATION_SCOPE", raising=False)
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    st = build_state(tmp_path, monkeypatch)
    key = attribution_text(st)
    res = crit_mod.critic(st)
    assert high_numbers(res, key) == []
    assert res["critic"]["config"]["citation_scope"] == "paragraph"


def test_sentence_scope_catches_a_number_taken_from_another_study(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_CITATION_SCOPE", "sentence")
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    st = build_state(tmp_path, monkeypatch)
    key = attribution_text(st)
    res = crit_mod.critic(st)
    bad = high_numbers(res, key)
    assert bad and "1018" in bad[0]["detail"] and res["needs_revision"]
    assert res["critic"]["config"]["citation_scope"] == "sentence"


def test_config_records_semantic_switch_and_defaults(tmp_path, monkeypatch):
    st = build_state(tmp_path, monkeypatch)
    monkeypatch.delenv("CRITIC_SEMANTIC", raising=False)
    cfg = crit_mod.critic(st)["critic"]["config"]
    assert cfg["semantic_llm_check"] is False and cfg["max_revisions"] == 2
    monkeypatch.setenv("CRITIC_SEMANTIC", "1")
    monkeypatch.setenv("CRITIC_MODEL", "my-critic")
    cfg = crit_mod.critic(st)["critic"]["config"]
    assert cfg["semantic_llm_check"] is True and cfg["critic_model"] == "my-critic"


def test_pooling_rule_needs_several_citations_or_meta_analysis_terms(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    st = build_state(tmp_path, monkeypatch)
    sids = [s["sid"] for s in st["synthesis"]["studies"]]
    block = next(b for b in st["synthesis"]["blocks"] if b["kind"] == "group")
    key = block["key"]
    s1, s2 = block["sids"][:2]
    block["text"] = f"The average age was reported in the cohort [{s1}]."          # single citation: not pooling
    assert not [i for i in crit_mod.critic(st)["critic"]["issues"] if i["section"] == key and i["kind"] == "pooling_language"]
    block["text"] = f"On average the models reached a high accuracy [{s1}, {s2}]."   # several studies: flagged
    assert [i for i in crit_mod.critic(st)["critic"]["issues"] if i["section"] == key and i["kind"] == "pooling_language"]
    block["text"] = f"A meta-analysis shows the effect [{s1}]."                       # explicit term: flagged
    assert [i for i in crit_mod.critic(st)["critic"]["issues"] if i["section"] == key and i["kind"] == "pooling_language"]
    block["text"] = f"No meta-analysis was performed [{s1}, {s2}]."                   # negated: fine
    assert not [i for i in crit_mod.critic(st)["critic"]["issues"] if i["section"] == key and i["kind"] == "pooling_language"]


@pytest.mark.parametrize("semantic,expect", [("0", "No LLM-based semantic check was run"),
                                             ("1", "compared with the studies it cites by an LLM critic")])
def test_report_describes_the_critic_as_it_really_ran(tmp_path, monkeypatch, semantic, expect):
    pytest.importorskip("langgraph")
    from src.graph import build_synthesis_graph
    monkeypatch.setenv("CRITIC_SEMANTIC", semantic)
    monkeypatch.setenv("CRITIC_CITATION_SCOPE", "paragraph")
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, T.make_chat())
    final = build_synthesis_graph().invoke(T.state_for(str(run)))
    md = open(final["report_path"], encoding="utf-8").read()
    assert expect in md and "anywhere in the same paragraph" in md
