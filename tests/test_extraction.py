import csv
import io
import json

import src.agents.extraction as ex
import src.tools.fulltext as ft
from src.rag.chunks import chunk_text, ChunkRetriever

PROTO = {"research_question": "q", "inclusion_criteria": ["a"], "exclusion_criteria": ["b", "c"],
         "extraction_fields": ["dataset", "sample_size", "metrics"]}


def make_pdf(text):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in text.split("\n"):
        c.drawString(40, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = 800
    c.save()
    return buf.getvalue()


def test_chunk_and_retrieve():
    body = ("Introduction text about nothing. " * 40 +
            "The dataset contained 1200 images from 3 hospitals. " + "Filler sentence here. " * 60 +
            "Accuracy reached 94 percent and AUC 0.97 on the test set. " + "More filler words. " * 40 +
            " References [1] A. Author.")
    chunks = chunk_text(body)
    assert all("References [1]" not in c for c in chunks)
    ret = ChunkRetriever(chunks)
    assert "AUC" in chunks[ret.top("accuracy AUC results", 1)[0]]


def test_fulltext_pdf_and_cache(tmp_path, monkeypatch):
    text = "\n".join(f"Line {i} of the methods section with enough words to count." for i in range(60))
    pdf = make_pdf(text)

    class R:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def iter_content(self, n):
            yield self.data

    calls = []
    monkeypatch.setattr(ft.requests, "get", lambda url, **k: (calls.append(url), R(pdf))[1])
    got, info = ft.fetch_fulltext({"id": "https://openalex.org/W1", "oa_pdf": "http://x/p.pdf"}, str(tmp_path))
    assert got and "methods section" in got and info["source_url"] == "http://x/p.pdf"
    got2, info2 = ft.fetch_fulltext({"id": "https://openalex.org/W1", "oa_pdf": "http://x/p.pdf"}, str(tmp_path))
    assert got2 == got and info2["source_url"] == "cache" and len(calls) == 1


def test_fulltext_html_page_is_rejected(tmp_path, monkeypatch):
    class R:
        status_code = 200

        def iter_content(self, n):
            yield b"<html>landing page</html>"

    monkeypatch.setattr(ft.requests, "get", lambda url, **k: R())
    got, info = ft.fetch_fulltext({"id": "w2", "oa_pdf": "http://x/landing"}, str(tmp_path))
    assert got is None and "pdf:not_a_pdf" in info["reason"]


def test_rob_overall():
    mk = lambda *r: {f"i{k}": {"rating": x} for k, x in enumerate(r)}
    assert ex.rob_overall(mk("low", "low")) == "low"
    assert ex.rob_overall(mk("low", "unclear")) == "some concerns"
    assert ex.rob_overall(mk("unclear", "high")) == "high"


def test_extraction_flow(tmp_path, monkeypatch):
    recs = [{"id": "a", "title": "A", "abstract": "abs a", "year": 2021, "doi": "10.1/a"},
            {"id": "b", "title": "B", "abstract": "abs b", "year": 2022, "doi": "10.1/b"},
            {"id": "c", "title": "C", "abstract": "abs c", "year": 2023, "doi": ""}]
    long_text = "The dataset has 500 images. " * 100
    monkeypatch.setattr(ex, "fetch_all", lambda r, d: {
        "a": (long_text, {"source_url": "u"}), "b": (None, {"reason": "no_open_access_link"}),
        "c": (None, {"reason": "not_a_pdf"})})

    def fake(system, user, **k):
        if "RECORD: B" in user:   # excluded at the full-text stage
            return {"eligibility": {"decision": "exclude", "reason_code": "E2", "rationale": "x"},
                    "fields": {}, "risk_of_bias": {}}
        if "RECORD: C" in user:   # messy output must be normalised
            return {"eligibility": {"decision": "include"}, "fields": {"dataset": "just a string"},
                    "risk_of_bias": {"selection_sampling": {"rating": "HIGH", "note": "n"},
                                     "method_description": {"rating": "bogus"}}}
        assert "[c" in user       # record A has full text: excerpts must be in the context
        return {"eligibility": {"decision": "include", "reason_code": "none", "rationale": "ok"},
                "fields": {"dataset": {"value": "500 images", "source": "c1"},
                           "sample_size": {"value": "500", "source": "c1"}},
                "risk_of_bias": {k: {"rating": "low", "note": "ok"} for k in ex.ROB_ITEMS}}

    monkeypatch.setattr(ex, "chat_json", fake)
    out = ex.extraction({"protocol": PROTO, "run_dir": str(tmp_path), "included": recs,
                         "prisma": {"included_for_fulltext": 3}})
    pr = out["prisma"]
    assert pr["fulltext_sought"] == 3 and pr["fulltext_retrieved"] == 1 and pr["assessed_on_abstract_only"] == 2
    assert pr["excluded_after_fulltext_stage"] == 1 and pr["included_in_review"] == 2
    assert list(pr["exclusion_reasons_fulltext_stage"]) == ["E2: c"]
    assert pr["risk_of_bias_overall"] == {"low": 1, "high": 1}
    rows = list(csv.DictReader(open(tmp_path / "evidence_table.csv", encoding="utf-8-sig")))
    assert len(rows) == 2 and rows[0]["dataset"] == "500 images" and rows[1]["dataset"] == "just a string"
    assert rows[1]["metrics"] == "not reported" and rows[1]["text_source"] == "abstract_only"
    assert json.load(open(tmp_path / "extraction.json"))[0]["text_source"] == "full_text"
