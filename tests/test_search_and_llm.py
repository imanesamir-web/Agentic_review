import src.agents.search_strategy as ss
import src.llm as llm


def test_lint():
    assert ss.lint_pubmed('("a"[MeSH] OR b[tiab]) AND c[tiab]') == []
    assert "unbalanced parentheses" in ss.lint_pubmed('(a AND b')
    assert any("Conference" in p for p in ss.lint_pubmed('NOT "Conference Abstract"[Publication Type]'))


def test_refinement_broadens_narrow_query(tmp_path, monkeypatch):
    replies = [{"pubmed": "(a[tiab]) AND (b[tiab]) AND (c[tiab])", "openalex": "a b c"},
               {"pubmed": "(a[tiab]) AND (b[tiab])", "openalex": "a b"}]
    prompts = []

    def fake(system, user, **k):
        prompts.append(user)
        return replies[len(prompts) - 1]

    monkeypatch.setattr(ss, "chat_json", fake)
    monkeypatch.setattr(ss.pubmed, "count", lambda q, a, b: 5 if q.count("AND") == 2 else 400)
    state = {"run_dir": str(tmp_path), "protocol": {
        "research_question": "q", "pico": {}, "inclusion_criteria": [], "exclusion_criteria": [],
        "max_records_per_database": 50, "date_range": {"from": 2015, "to": 2026}}}
    out = ss.search_strategy(state)
    assert out["queries"]["pubmed"] == "(a[tiab]) AND (b[tiab])"
    assert "too narrow" in prompts[1]


def test_cooldown_skips_rate_limited_model(monkeypatch):
    llm._cool.clear()
    monkeypatch.setenv("LLM_MODEL", "m1")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "m2")
    calls = []

    class R:
        def __init__(self, code):
            self.status_code, self.text = code, "x"

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls.append(json["model"])
        return R(429 if json["model"] == "m1" else 200)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.chat("s", "u") == "ok" and calls == ["m1", "m2"]
    assert llm.chat("s", "u") == "ok" and calls == ["m1", "m2", "m2"]  # m1 skipped (cooling)
    llm._cool.clear()


def test_search_strategy_normalizes_nested_query_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "chat_json", lambda *a, **k: {
        "pubmed": {"query": "(a[tiab]) AND (b[tiab])"},
        "openalex": {"search_query": "medical image classification"},
    })
    monkeypatch.setattr(ss.pubmed, "count", lambda q, a, b: 400)
    state = {"run_dir": str(tmp_path), "protocol": {
        "research_question": "q", "pico": {}, "inclusion_criteria": [], "exclusion_criteria": [],
        "max_records_per_database": 50, "date_range": {"from": 2015, "to": 2026}}}
    result = ss.search_strategy(state)
    assert result["queries"] == {
        "pubmed": "(a[tiab]) AND (b[tiab])",
        "openalex": "medical image classification",
    }


def test_lint_pubmed_rejects_non_text_without_crashing():
    assert ss.lint_pubmed({"query": "x"}) == ["query must be text"]
