import csv
import json

import src.agents.critic as crit_mod
import src.agents.synthesis as syn_mod
import src.evaluation as ev
import test_synthesis_stack as T


def build_state(tmp_path, monkeypatch, chat=None):
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, chat or T.make_chat())
    st = T.state_for(str(run))
    st.update(syn_mod.synthesis(st))
    return st, run


def group_block(st):
    return next(b for b in st["synthesis"]["blocks"] if b["kind"] == "group")


def pooling(res, key):
    return [i for i in res["critic"]["issues"] if i["section"] == key and i["kind"] == "pooling_language"]


def test_same_study_cited_twice_is_not_several_studies(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    monkeypatch.setenv("CRITIC_CITATION_SCOPE", "inherit")
    st, _ = build_state(tmp_path, monkeypatch)
    b = group_block(st)
    a, c = b["sids"][:2]
    b["text"] = f"Study [{a}] reported a mean AUC of 0.95 and mean accuracy of 0.91 [{a}]."   # one study, cited twice
    assert not pooling(crit_mod.critic(st), b["key"])
    for scope in ("sentence", "paragraph"):
        monkeypatch.setenv("CRITIC_CITATION_SCOPE", scope)
        assert not pooling(crit_mod.critic(st), b["key"]), scope
    b["text"] = f"The mean accuracy of the models was high [{a}, {c}]."                        # two different studies
    monkeypatch.setenv("CRITIC_CITATION_SCOPE", "inherit")
    assert pooling(crit_mod.critic(st), b["key"])


def test_llm_critic_sees_the_whole_metrics_field(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SEMANTIC", "1")
    st, _ = build_state(tmp_path, monkeypatch)
    b = group_block(st)
    sid = b["sids"][0]
    study = next(s for s in st["synthesis"]["studies"] if s["sid"] == sid)
    study["fields"]["metrics"] = "accuracy 91.2% " + "x" * 300 + " sensitivity 97.8%"      # 97.8 sits beyond 200 chars
    b["text"] = f"Study {sid} reported a sensitivity of 97.8% [{sid}]."
    seen = []
    monkeypatch.setattr(crit_mod, "chat_json", lambda s, u, **k: (seen.append(u), {"issues": [], "verdict": "pass"})[1])
    crit_mod.semantic_check(b, {x["sid"]: x for x in st["synthesis"]["studies"]}, T.PROTO)
    assert "sensitivity 97.8%" in seen[0]
    monkeypatch.setenv("CRITIC_EVIDENCE_CHARS", "200")
    crit_mod.semantic_check(b, {x["sid"]: x for x in st["synthesis"]["studies"]}, T.PROTO)
    assert "sensitivity 97.8%" not in seen[1]                                               # the old, truncated view


def test_critic_review_sheet_and_precision(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SEMANTIC", "1")
    monkeypatch.setenv("CRITIC_LLM_REWRITES", "0")
    monkeypatch.setenv("MAX_REVISIONS", "0")
    st, run = build_state(tmp_path, monkeypatch, T.make_chat(bad_group="Ultrasound"))

    def fake(system, user, **k):
        text = user.split("TEXT:\n")[1].split("\n\nEVIDENCE")[0]
        first = text.split(". ")[0].rstrip(".") + "."
        return {"issues": [{"sentence": first, "problem": "p-high", "severity": "high"},
                           {"sentence": first, "problem": "p-medium", "severity": "medium"}], "verdict": "revise"}

    monkeypatch.setattr(crit_mod, "chat_json", fake)
    crit_mod.critic(st)
    sheet, sizes = ev.critic_review_sheet(str(run), n=9)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    assert set(sizes) == {"llm/high", "llm/medium", "deterministic/high"} and 6 <= len(rows) <= 9
    assert "Study" in rows[0]["evidence"] or "[S" in rows[0]["evidence"]
    for j, r in enumerate(rows):
        r["human_verdict (valid/invalid/unsure)"] = "valid" if j % 2 == 0 else "invalid"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = ev.score_critic_review(str(run))["strata"]
    key = json.load(open(run / "eval" / "critic_review_key.json"))["rows"]
    for stratum, v in res.items():
        mine = [j for j, r in enumerate(rows) if key[str(r["id"])] == stratum]
        valid = sum(1 for j in mine if j % 2 == 0)
        assert v["valid"] == valid and v["judged"] == len(mine)
        assert v["precision"] == round(valid / len(mine), 3)
    assert "precision" in ev.format_critic_review({"strata": res, "note": "n"})


def test_report_states_citation_coverage(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("langgraph")
    from src.graph import build_synthesis_graph
    monkeypatch.setenv("CRITIC_SEMANTIC", "0")
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, T.make_chat())
    final = build_synthesis_graph().invoke(T.state_for(str(run)))
    md = open(final["report_path"], encoding="utf-8").read()
    assert "Citation coverage:" in md and "sentences in the group passages carry a citation of their own" in md
