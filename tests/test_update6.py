import csv
import json
import os

import src.agents.extraction as ex
import src.evaluation as ev
import src.fileutil as fu
import src.timing as tm
from src.agents.extraction import ROB_ITEMS

PROTO = {"research_question": "q", "inclusion_criteria": ["a"], "exclusion_criteria": ["b"],
         "extraction_fields": ["dataset", "metrics"]}


def srow(i, final, a=None, b=None):
    mk = lambda d: {"decision": d, "reason_code": "none", "reason": "", "rationale": ""}
    return {"id": i, "title": f"T{i}", "year": 2020, "reviewer_a": mk(a or final), "reviewer_b": mk(b or final),
            "agreement": True, "final": mk(final)}


def make_run(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    rows = [srow(f"i{k}", "include") for k in range(20)] + [srow(f"e{k}", "exclude") for k in range(10)]
    json.dump(rows, open(d / "screening.json", "w"))
    json.dump([{"id": r["id"], "abstract": f"abs {r['id']}"} for r in rows], open(d / "records.json", "w"))
    json.dump(PROTO, open(d / "protocol.json", "w"))
    return str(d)


def test_screening_sheet_is_blind_and_balanced(tmp_path):
    run = make_run(tmp_path)
    sheet, key = ev.make_screening_sheet(run, n=10)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    assert len(rows) == 10 and (key["sampled_include"], key["sampled_exclude"]) == (5, 5)
    assert set(rows[0]) == {"id", "title", "year", "abstract", "human_decision", "notes"}   # no agent decision shown
    assert os.path.exists(f"{run}/eval/criteria.txt")


def test_screening_score_reweights_strata(tmp_path):
    run = make_run(tmp_path)
    sheet, key = ev.make_screening_sheet(run, n=10)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    flipped = False
    for r in rows:      # you agree with the pipeline except for ONE included record you would exclude
        if r["id"].startswith("i") and not flipped:
            r["human_decision"], flipped = "exclude", True
        else:
            r["human_decision"] = "include" if r["id"].startswith("i") else "exclude"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = ev.score_screening(run)["predictors"]["pipeline_final"]
    # weights: include stratum 20/5 = 4, exclude stratum 10/5 = 2 -> TP 16, FP 4, TN 10, FN 0
    assert res["raw_counts"] == {"tp": 4, "fp": 1, "tn": 5, "fn": 0}
    m = res["weighted"]
    assert (m["sensitivity"], m["precision"]) == (1.0, 0.8)
    assert abs(m["specificity"] - 0.714) < 1e-3 and abs(m["accuracy"] - 0.867) < 1e-3


def test_extraction_changes_vs_audit_trail(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    f = lambda v: {"value": v, "source": "abstract"}
    items = lambda r: {k: {"rating": r, "note": ""} for k in ROB_ITEMS}
    orig = {"id": "a", "title": "A", "text_source": "full_text", "extraction_failed": False,
            "eligibility": {"decision": "include"}, "fields": {"dataset": f("BUSI"), "metrics": f("accuracy 90%")},
            "risk_of_bias": items("low"), "overall_rob": "low"}
    failed = dict(orig, id="b", extraction_failed=True)
    with open(d / "decisions.jsonl", "w") as fh:
        for row in (failed, orig):
            fh.write(json.dumps({"stage": "extraction", "event": "record_processed", "data": row}) + "\n")
    cur = json.loads(json.dumps(orig))
    cur["fields"]["metrics"] = f("accuracy 92.5%")          # you corrected one field
    cur["fields"]["dataset"] = f("busi.")                    # cosmetic difference only: not an edit
    cur["risk_of_bias"]["analysis_validation"]["rating"] = "high"
    cur["overall_rob"] = "high"
    json.dump([cur], open(d / "extraction.json", "w"))
    res = ev.extraction_changes(str(d))
    assert res["records_compared"] == 1 and res["field_edit_rate_by_field"]["metrics"]["edit_rate"] == 1.0
    assert res["field_edit_rate_by_field"]["dataset"]["edit_rate"] == 0.0
    assert res["field_edit_rate_overall"] == 0.5 and res["rob_overall_agreement"] == 0.0
    assert res["rob_item_agreement"]["analysis_validation"]["agreement"] == 0.0


def test_timing_wrapper_and_time_report(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    fn = tm.timed("screening", lambda s: {"x": 1})
    assert fn({"run_dir": str(d)}) == {"x": 1}
    tm.record(str(d), "screening", 60.0)
    t = json.load(open(d / "timing.json"))["screening"]
    assert t["calls"] == 2 and t["seconds"] >= 60.0
    json.dump({"screened_title_abstract": 100, "disagreements_arbitrated": 2, "fulltext_sought": 78},
              open(d / "prisma_counts.json", "w"))
    res = ev.time_report(str(d))
    scr = next(r for r in res["rows"] if r["stage"].startswith("Title/abstract"))
    assert scr["manual_estimate_min"] == 210.0 and scr["automated_min"] is not None
    assert next(r for r in res["rows"] if r["stage"].startswith("Full text"))["automated_min"] is None
    assert os.path.exists(d / "eval" / "time_assumptions.json")


def test_safe_path_when_locked(tmp_path, monkeypatch):
    p = str(tmp_path / "evidence_table.csv")
    assert fu.safe_path(p) == p
    monkeypatch.setattr(fu, "_can_write", lambda path: False)
    alt = fu.safe_path(p)
    assert alt != p and alt.endswith(".csv")


GOOD = {"eligibility": {"decision": "include", "reason_code": "none", "rationale": "ok"},
        "fields": {"dataset": {"value": "D", "source": "abstract"}, "metrics": {"value": "m", "source": "abstract"}},
        "risk_of_bias": {k: {"rating": "low", "note": ""} for k in ROB_ITEMS}}


def _prev_row(i, **kw):
    r = {"id": i, "title": i.upper(), "year": 2020, "doi": "", "text_source": "abstract_only", "fulltext_info": {},
         "extraction_failed": False, **GOOD, "overall_rob": "low", **kw}
    r["fields"] = {"dataset": {"value": f"manual-{i}", "source": "abstract"}, "metrics": {"value": "m", "source": "abstract"}}
    return r


def test_curated_rows_survive_force_and_backup_is_made(tmp_path, monkeypatch):
    recs = [{"id": i, "title": i.upper(), "abstract": "x", "year": 2020} for i in "ab"]
    run = tmp_path / "run"
    run.mkdir()
    json.dump([_prev_row("a", curated=True), _prev_row("b")], open(run / "extraction.json", "w"))
    monkeypatch.setenv("EXTRACT_FORCE", "1")
    monkeypatch.setattr(ex, "fetch_all", lambda r, d: {"a": ("T " * 2000, {}), "b": ("T " * 2000, {})})
    calls = []
    monkeypatch.setattr(ex, "chat_json", lambda s, u, **k: (calls.append(u.split("RECORD: ")[1][0]), GOOD)[1])
    ex.extraction({"protocol": PROTO, "run_dir": str(run), "included": recs, "prisma": {}})
    assert calls == ["B"]                                  # curated record A was NOT re-extracted, even with FORCE
    rows = {r["id"]: r for r in json.load(open(run / "extraction.json"))}
    assert rows["a"]["fields"]["dataset"]["value"] == "manual-a"
    assert len(os.listdir(run / "backups")) == 1


def test_csv_locked_does_not_crash_extraction(tmp_path, monkeypatch):
    recs = [{"id": "a", "title": "A", "abstract": "x", "year": 2020}]
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(ex, "fetch_all", lambda r, d: {"a": (None, {})})
    monkeypatch.setattr(ex, "chat_json", lambda s, u, **k: GOOD)
    monkeypatch.setattr(fu, "_can_write", lambda path: not path.endswith("evidence_table.csv"))
    ex.extraction({"protocol": PROTO, "run_dir": str(run), "included": recs, "prisma": {}})
    assert (run / "extraction.json").exists() and (run / "prisma_counts.json").exists()
    assert any(n.startswith("evidence_table_") for n in os.listdir(run))


def test_reuse_skips_the_narrative_and_runs_critic_and_report(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("langgraph")
    import test_synthesis_stack as T
    import src.agents.synthesis as syn_mod
    from src.graph import build_synthesis_graph
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, T.make_chat())
    st = T.state_for(str(run))
    st.update(syn_mod.synthesis(st))                      # first run: writes the narrative
    narrative_calls = []
    monkeypatch.setattr(syn_mod, "chat", lambda *a, **k: narrative_calls.append(1))
    final = build_synthesis_graph("critic").invoke({**T.state_for(str(run)), "synthesis": st["synthesis"]})
    assert narrative_calls == [] and os.path.exists(final["report_path"])
    assert (run / "timing.json").exists() and "critic" in json.load(open(run / "timing.json"))
