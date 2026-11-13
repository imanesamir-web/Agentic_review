import json

import pytest

import src.agents.extraction as ex
import src.llm as llm
import src.tools.fulltext as ft

JATS = (b'<?xml version="1.0" encoding="UTF-8"?><article><front><article-meta><title-group>'
        b'<article-title>T</article-title></title-group></article-meta></front><body><sec><title>Methods</title>'
        b'<p>We used <italic>500</italic> images from two hospitals.</p></sec>'
        b'<sec><title>Results</title><p>AUC was 0.95.</p></sec></body>'
        b'<back><ref-list><ref><p>Should not appear.</p></ref></ref-list></back></article>')


def test_jats_to_text():
    t = ft.jats_to_text(JATS)
    assert "Methods We used 500 images from two hospitals." in t and "AUC was 0.95." in t
    assert "Should not appear" not in t


class Resp:
    def __init__(self, status=200, js=None, content=b""):
        self.status_code, self._js, self.content, self.text = status, js, content, ""

    def json(self):
        return self._js


def _big_jats():
    return JATS.replace(b"AUC was 0.95.", b"AUC was 0.95. " + b"More results text. " * 200)


def test_epmc_success(monkeypatch):
    seen = []

    def fake_get(url, **k):
        seen.append(url)
        if url.endswith("/search"):
            assert k["params"]["query"] == "EXT_ID:123 AND SRC:MED"
            return Resp(js={"resultList": {"result": [{"pmcid": "PMC9", "isOpenAccess": "Y"}]}})
        return Resp(content=_big_jats())

    monkeypatch.setattr(ft.requests, "get", fake_get)
    text, src = ft.epmc_fulltext({"id": "pmid:123", "doi": ""})
    assert text and src.endswith("/PMC9/fullTextXML")


def test_epmc_not_open_access_and_doi_query(monkeypatch):
    def fake_get(url, **k):
        assert k["params"]["query"] == 'DOI:"10.1/x"'
        return Resp(js={"resultList": {"result": [{"pmcid": "PMC9", "isOpenAccess": "N"}]}})

    monkeypatch.setattr(ft.requests, "get", fake_get)
    text, reason = ft.epmc_fulltext({"id": "https://openalex.org/W1", "doi": "10.1/x"})
    assert text is None and reason == "epmc:not_open_access"


def test_fetch_uses_epmc_and_caches(tmp_path, monkeypatch):
    def fake_get(url, **k):
        if url.endswith("/search"):
            return Resp(js={"resultList": {"result": [{"pmcid": "PMC9", "isOpenAccess": "Y"}]}})
        return Resp(content=_big_jats())

    monkeypatch.setattr(ft.requests, "get", fake_get)
    rec = {"id": "pmid:123", "doi": ""}
    text, info = ft.fetch_fulltext(rec, str(tmp_path))
    assert text and "fullTextXML" in info["source_url"]
    assert ft.fetch_fulltext(rec, str(tmp_path))[1]["source_url"] == "cache"


PROTO = {"research_question": "q", "inclusion_criteria": ["a"], "exclusion_criteria": ["b"],
         "extraction_fields": ["dataset"]}
GOOD = {"eligibility": {"decision": "include", "reason_code": "none", "rationale": "ok"},
        "fields": {"dataset": {"value": "d", "source": "abstract"}},
        "risk_of_bias": {k: {"rating": "low", "note": ""} for k in ex.ROB_ITEMS}}


def test_resume_only_redoes_failed_or_newly_fulltext(tmp_path, monkeypatch):
    recs = [{"id": i, "title": i.upper(), "abstract": "x", "year": 2020} for i in "abc"]
    monkeypatch.setattr(ex, "fetch_all", lambda r, d: {
        "a": ("T " * 2000, {"source_url": "u"}),   # was abstract-only, full text found now -> redo
        "b": (None, {"reason": "x"}),               # failed before -> redo
        "c": (None, {"reason": "x"})})              # fine before, still abstract only -> reuse
    calls = []
    monkeypatch.setattr(ex, "chat_json", lambda s, u, **k: (calls.append(u.split("RECORD: ")[1][0]), GOOD)[1])

    def prev_row(i, failed):
        if failed:
            return ex._failed_row(recs["abc".index(i)], PROTO, None, {})
        return {"id": i, "title": i.upper(), "year": 2020, "doi": "", "text_source": "abstract_only",
                "fulltext_info": {}, "extraction_failed": False,
                "eligibility": GOOD["eligibility"] | {"reason": ""},
                "fields": {"dataset": {"value": "old", "source": "abstract"}},
                "risk_of_bias": GOOD["risk_of_bias"], "overall_rob": "low"}

    json.dump([prev_row("a", False), prev_row("b", True), prev_row("c", False)],
              open(tmp_path / "extraction.json", "w"))
    out = ex.extraction({"protocol": PROTO, "run_dir": str(tmp_path), "included": recs, "prisma": {}})
    assert sorted(calls) == ["A", "B"]
    pr = out["prisma"]
    assert pr["extraction_reused_from_previous_run"] == 1 and pr["extraction_pending_retry"] == 0
    assert pr["risk_of_bias_overall"] == {"low": 3}


def test_failed_records_are_pending_not_rated(tmp_path, monkeypatch):
    recs = [{"id": "a", "title": "A", "abstract": "x", "year": 2020}]
    monkeypatch.setattr(ex, "fetch_all", lambda r, d: {"a": (None, {})})

    def boom(*a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr(ex, "chat_json", boom)
    pr = ex.extraction({"protocol": PROTO, "run_dir": str(tmp_path), "included": recs, "prisma": {}})["prisma"]
    assert pr["extraction_pending_retry"] == 1 and pr["risk_of_bias_overall"] == {}
    assert pr["included_in_review"] == 1


def test_json_mode_payload_and_quota_error(monkeypatch):
    llm._cool.clear()
    monkeypatch.setenv("LLM_MODEL", "m1")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")
    monkeypatch.setenv("LLM_JSON_MODE", "1")
    sent = []

    class R:
        status_code, text = 200, ""

        def json(self):
            return {"choices": [{"message": {"content": '{"a": 1}'}}]}

    monkeypatch.setattr(llm.requests, "post", lambda url, headers=None, json=None, timeout=0: (sent.append(json), R())[1])
    assert llm.chat_json("s", "u") == {"a": 1}
    assert sent[0]["response_format"] == {"type": "json_object"}

    class Q:
        status_code, text = 429, '{"error":{"message":"Rate limit exceeded: free-models-per-day"}}'

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Q())
    with pytest.raises(RuntimeError, match="daily free-model quota"):
        llm.chat("s", "u")
    llm._cool.clear()
