import json
from src.metrics import cohen_kappa
import src.agents.screening as sc

PROTO = {"research_question": "q", "inclusion_criteria": ["a"], "exclusion_criteria": ["b", "c"]}


def test_kappa():
    assert cohen_kappa(["include"] * 4, ["include"] * 4) == 1.0
    k = cohen_kappa(["include", "exclude", "include", "exclude"],
                    ["include", "exclude", "exclude", "exclude"])
    assert abs(k - 0.5) < 1e-9


def test_norm_code():
    assert sc.norm_code("E2", 2) == "E2"
    assert sc.norm_code(" e 1 ", 2) == "E1"
    assert sc.norm_code("E7", 2) == "E0"
    assert sc.norm_code("Review article", 2) == "none"


def test_screening_flow(tmp_path, monkeypatch):
    recs = [{"id": f"r{i}", "title": f"T{i}", "abstract": "x", "year": 2020} for i in range(7)]

    def fake(system, user, model=None, temperature=0.0, retries=3):
        if "arbiter" in system:
            return {"decision": "exclude", "reason_code": "E2", "rationale": "arb"}
        dec = []
        for i in range(1, 6):
            if f"[{i}] Title" in user:
                title_line = user.split(f"[{i}] Title: ")[1].split("\n")[0]
                d, code = "include", "none"
                if title_line == "T3" and temperature > 0:
                    d, code = "exclude", "E1"
                dec.append({"id": str(i), "decision": d, "reason_code": code, "rationale": "ok"})
        return {"decisions": dec}

    monkeypatch.setattr(sc, "chat_json", fake)
    state = {"protocol": PROTO, "run_dir": str(tmp_path), "records": recs,
             "prisma": {"records_after_dedup": 7}}
    pr = sc.screening(state)["prisma"]
    assert pr["screened_title_abstract"] == 7 and pr["disagreements_arbitrated"] == 1
    assert pr["included_for_fulltext"] == 6 and pr["excluded_title_abstract"] == 1
    assert list(pr["exclusion_reasons"]) == ["E2: c"]  # arbiter's coded reason, normalised
    rows = json.load(open(tmp_path / "screening.json"))
    assert rows[3]["final"]["by"] == "arbiter"


def test_agreed_exclusion_uses_reviewer_code(tmp_path, monkeypatch):
    recs = [{"id": "x", "title": "T", "abstract": "y", "year": 2021}]
    monkeypatch.setattr(sc, "chat_json", lambda *a, **k: {"decisions": [
        {"id": "1", "decision": "exclude", "reason_code": "E1", "rationale": "r"}]})
    pr = sc.screening({"protocol": PROTO, "run_dir": str(tmp_path), "records": recs})["prisma"]
    assert pr["exclusion_reasons"] == {"E1: b": 1} and pr["disagreements_arbitrated"] == 0


def test_missing_decisions_are_maybe(monkeypatch):
    monkeypatch.setattr(sc, "chat_json", lambda *a, **k: {"decisions": []})
    out = sc.screen_batch([{"id": "z", "title": "t", "abstract": ""}], PROTO)
    assert out["z"]["decision"] == "maybe" and out["z"]["reason"] == "screening_error"
