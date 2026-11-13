import json
import re

import pytest

import src.agents.critic as crit_mod
import src.agents.synthesis as syn_mod
import src.evaluation as ev
import test_synthesis_stack as T


def worse_on_rewrite_chat():
    base = T.make_chat()

    def chat(system, user, **k):
        if "Results synthesis" in system:
            grp = re.search(r"^Group: (.+)$", user, re.M).group(1)
            text = T.clean_group_text(user)
            if grp == "Ultrasound":
                if "PREVIOUS DRAFT" in user:     # the 'improved' text is worse: three unsupported numbers
                    text += " One reached 99.9% [S01]. Another reached 88.8% [S01]. A third reached 77.7% [S01]."
                else:                            # first draft: one unsupported number
                    text += " One study reached an accuracy of 99.9% [S01]."
            return text
        return base(system, user, **k)

    return chat


def make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    return run


def test_a_rewrite_that_is_not_better_is_rejected_and_the_best_text_is_kept(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    from src.graph import build_synthesis_graph
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    monkeypatch.setenv("MAX_REVISIONS", "2")
    run = make_run(tmp_path)
    T.patch(monkeypatch, worse_on_rewrite_chat())
    final = build_synthesis_graph().invoke(T.state_for(str(run)))
    crit = final["critic"]
    assert crit["verdict"] == "pass_with_unresolved_issues" and crit["best_pass"] == 1
    assert len(crit["history"]) == 2 and "rewrite rejected" in crit["history"][1]["note"]
    text = next(b["text"] for b in final["synthesis"]["blocks"] if b["group"] == "Ultrasound")
    assert "99.9%" in text and "88.8" not in text                      # first draft kept, not the worse rewrite
    assert "88.8" not in open(run / "synthesis.json", encoding="utf-8").read()   # and not left on disk
    md = open(final["report_path"], encoding="utf-8").read()
    assert "rewrite rejected" in md and "⚠[unverified]" in md and "88.8" not in md


def test_critic_summary_lists_checks_and_examples(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    run = make_run(tmp_path)
    T.patch(monkeypatch, T.make_chat(bad_group="Ultrasound"))
    st = T.state_for(str(run))
    st.update(syn_mod.synthesis(st))
    crit_mod.critic(st)
    res = ev.critic_summary(str(run))
    checks = {(r["severity"], r["check"]) for r in res["table"]}
    assert ("high", "number_unsupported") in checks and ("high", "pooling_language") in checks
    assert res["high_examples"]["number_unsupported"][0]["sentence"]
    assert "number_unsupported" in ev.format_critic_summary(res)
