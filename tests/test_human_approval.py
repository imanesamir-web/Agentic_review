import builtins
import json
import os
from pathlib import Path

import src.agents.planner as planner_mod
import src.agents.synthesis as syn_mod
import test_synthesis_stack as T

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {"title": "t", "pico": {}, "study_design": [], "inclusion_criteria": ["a"], "exclusion_criteria": ["b"]}


def test_protocol_approval_picks_up_the_human_edit(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("HUMAN_APPROVAL", "1")
    monkeypatch.setattr(planner_mod, "chat_json", lambda s, u, **k: json.loads(json.dumps(GENERATED)))
    run = tmp_path / "run"
    run.mkdir()

    def human(prompt=""):
        path = run / "protocol.json"
        proto = json.load(open(path, encoding="utf-8"))
        proto["inclusion_criteria"].append("added by the human")
        json.dump(proto, open(path, "w", encoding="utf-8"))
        return ""

    monkeypatch.setattr(builtins, "input", human)
    out = planner_mod.planner({"run_dir": str(run), "question": "q?"})
    assert out["protocol"]["inclusion_criteria"] == ["a", "added by the human"]
    assert '"edited": true' in open(run / "decisions.jsonl", encoding="utf-8").read()


def test_protocol_approval_off_by_default_never_asks(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("HUMAN_APPROVAL", raising=False)
    monkeypatch.setattr(planner_mod, "chat_json", lambda s, u, **k: json.loads(json.dumps(GENERATED)))
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(AssertionError("must not ask")))
    run = tmp_path / "run"
    run.mkdir()
    assert planner_mod.planner({"run_dir": str(run), "question": "q?"})["protocol"]["inclusion_criteria"] == ["a"]


def test_grouping_approval_honours_the_edited_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMAN_APPROVAL", "1")
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, T.make_chat())

    def human(prompt=""):
        path = run / "group_assignments.json"
        ga = json.load(open(path, encoding="utf-8"))
        ga["assignments"]["b2"]["group"] = "CT"          # move the MRI study to CT
        json.dump(ga, open(path, "w", encoding="utf-8"))
        return ""

    monkeypatch.setattr(builtins, "input", human)
    syn = syn_mod.synthesis(T.state_for(str(run)))["synthesis"]
    assert next(s for s in syn["studies"] if s["id"] == "b2")["group"] == "CT"
    assert "MRI" not in {g["label"] for g in syn["groups"]}
