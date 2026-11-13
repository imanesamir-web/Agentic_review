import json
import os

import src.agents.synthesis as syn_mod
import src.evaluation as ev
import test_synthesis_stack as T


def test_canary_three_scopes_and_two_controls(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    json.dump(T.ROWS, open(run / "extraction.json", "w"))
    json.dump(T.RECORDS, open(run / "included_records.json", "w"))
    T.patch(monkeypatch, T.make_chat())
    syn_mod.synthesis(T.state_for(str(run)))
    monkeypatch.setenv("CRITIC_CITATION_SCOPE", "paragraph")
    res = ev.critic_canary(str(run))
    by = {r["seeded_error"]: r for r in res["cases"]}
    for label in ("fabricated number", "pooling across studies", "citation of a non-existent study",
                  "overstated claim", "citation from another group"):
        assert all(by[label][sc] for sc in ("paragraph", "inherit", "sentence")), label
    attr = by["number taken from another study in the same paragraph"]
    assert (attr["paragraph"], attr["inherit"], attr["sentence"]) == (False, True, True)
    assert res["detected"] == {"paragraph": "5/6", "inherit": "6/6", "sentence": "6/6"}
    cited, late = "faithful, cited", "faithful, but citation only at the end of the paragraph"
    for sc in ("paragraph", "inherit", "sentence"):
        assert res["controls"][sc][cited] == []
    assert res["controls"]["paragraph"][late] == [] and res["controls"]["inherit"][late] == []
    assert res["controls"]["sentence"][late] == ["number_unsupported"]          # the strict scope's false alarm
    assert res["false_alarms_on_faithful_text"] == {"paragraph": 0, "inherit": 0, "sentence": 1}
    assert os.environ["CRITIC_CITATION_SCOPE"] == "paragraph"
    out = ev.format_canary(res)
    assert "MISSED" in out and "inherit" in out
