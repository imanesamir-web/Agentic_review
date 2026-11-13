import csv
import json

import src.evaluation as ev
import src.timing as tm
import test_synthesis_stack as T
import test_update6 as U6


def fill(path, col, values_for_rows):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    for j, r in enumerate(rows):
        r[col] = values_for_rows(j, r)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def labelled_run(tmp_path):
    run = U6.make_run(tmp_path)
    sheet, key = ev.make_screening_sheet(run, n=10)
    # you accept every pipeline inclusion, but include 4 of the 5 records the pipeline excluded
    seen_exclude = []

    def label(j, r):
        if r["id"].startswith("i"):
            return "include"
        seen_exclude.append(r["id"])
        return "exclude" if len(seen_exclude) == 1 else "include"

    fill(sheet, "human_decision", label)
    return run


def test_screening_intervals_disagreements_and_determinism(tmp_path):
    run = labelled_run(tmp_path)
    res = ev.score_screening(run)
    p = res["predictors"]["pipeline_final"]
    assert p["raw_counts"] == {"tp": 5, "fp": 0, "tn": 1, "fn": 4}
    m, ci = p["weighted"], p["ci95"]
    for k in ("sensitivity", "precision", "accuracy"):
        assert ci[k][0] <= m[k] <= ci[k][1], k
    assert m["specificity"] == 1.0 and ci["specificity"][0] < 0.5       # one true negative: a wide interval, not [1, 1]
    assert ci["precision"][0] < 1.0 and ci["sensitivity"][1] - ci["sensitivity"][0] > 0.1   # zero-error strata still uncertain
    assert "Jeffreys" in p["ci_method"]
    assert ev.score_screening(run)["predictors"]["pipeline_final"]["ci95"] == ci
    out = ev.screening_disagreements(run)
    assert len(out) == 4 and all(i["pipeline"] == "exclude" and i["human"] == "include" for i in out)
    text = ev.format_disagreements(out)
    assert "4 disagreements" in text and "95% intervals" in ev.format_screening(res)


def test_extraction_verification_sample_and_scores(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "records.json", "w"))
    sheet, info = ev.extraction_sheet(str(run), n=4)
    assert info == {"records": 4, "full_text": 1, "abstract_only": 3, "judgements": 20}
    verdicts = ["correct", "wrong", "partial", "unverifiable"]
    rows = fill(sheet, "verdict (correct/partial/wrong/unverifiable)", lambda j, r: verdicts[j % 4])
    res = ev.score_extraction_sheet(str(run))
    o = res["overall"]
    counts = {v: sum(1 for j in range(20) if verdicts[j % 4] == v) for v in verdicts}
    assert o["judged"] == counts["correct"] + counts["partial"] + counts["wrong"] == 15
    assert o["accuracy_strict"] == round(counts["correct"] / 15, 3)
    lo, hi = o["accuracy_strict_ci95"]
    assert 0 <= lo <= o["accuracy_strict"] <= hi <= 1
    assert set(res["by_text_source"]) == {"full_text", "abstract_only"} and set(res["by_field"]) == set(ev.VERIFY_FIELDS)
    assert "OVERALL" in ev.format_extraction_verify(res)


def test_wilson_interval():
    lo, hi = ev.wilson(8, 10)
    assert 0.48 < lo < 0.50 and 0.93 < hi < 0.95 and ev.wilson(0, 0) == [None, None]


def test_time_report_warns_when_a_stage_ran_several_times(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    tm.record(str(d), "screening", 60.0)
    tm.record(str(d), "screening", 60.0)
    json.dump({"screened_title_abstract": 100, "disagreements_arbitrated": 2, "fulltext_sought": 78}, open(d / "prisma_counts.json", "w"))
    res = ev.time_report(str(d))
    scr = next(r for r in res["rows"] if r["stage"].startswith("Title/abstract"))
    assert scr["runs"] == 2 and "(2 runs!)" in ev.format_time(res) and "ADDS UP" in res["note"]
