import json

import src.agents.search_strategy as ss

PROTO = {"research_question": "Medical image classification using deep learning", "pico": {},
         "inclusion_criteria": [], "exclusion_criteria": [], "max_records_per_database": 50,
         "date_range": {"from": 2015, "to": 2026}}

# the kind of query llama3.1:8b produced in the clean run
BAD = ("((('Medical images'[Title/Abstract] OR 'Deep learning'[MeSH Terms]) AND ('Accuracy'[MeSH Terms])) "
       "AND English[Language] AND NOT ('Conference papers'[Publication Type] OR 'Articles without DOI'[Title/Abstract])")


def run(tmp_path, monkeypatch, replies, count):
    seq = iter(replies)
    monkeypatch.setattr(ss, "chat_json", lambda s, u, **k: next(seq))
    monkeypatch.setattr(ss.pubmed, "count", count)
    return ss.search_strategy({"protocol": PROTO, "run_dir": str(tmp_path)}), json.load(open(tmp_path / "queries.json"))


def test_normaliser_repairs_single_quotes_and_and_not():
    q = ss.normalize_pubmed_query("('Deep learning'[MeSH Terms]) AND NOT ('Review articles'[Publication Type])")
    assert q == '("Deep learning"[MeSH Terms]) NOT ("Review articles"[Publication Type])'
    assert ss.lint_pubmed(q) == []


def test_keyword_fallback_is_always_valid():
    q = ss.keyword_pubmed_query(PROTO["research_question"])
    assert q == "Medical[tiab] AND image[tiab] AND classification[tiab] AND deep[tiab] AND learning[tiab]"
    assert ss.lint_pubmed(q) == [] and ss.lint_pubmed(ss.keyword_pubmed_query("???")) == []


def test_unusable_model_queries_trigger_the_fallback(tmp_path, monkeypatch):
    # three attempts: empty, then two queries with unbalanced parentheses
    replies = [{"pubmed": "", "openalex": "a b"}, {"pubmed": "(a[tiab] AND b[tiab]", "openalex": "a b"},
               {"pubmed": "((a[tiab]) AND b[tiab]", "openalex": "a b"}]
    out, saved = run(tmp_path, monkeypatch, replies, lambda q, a, b: 800)
    assert out["queries"]["pubmed"].startswith("Medical[tiab]") and saved["fallback_used"] is True
    assert saved["refinement_history"][-1]["attempt"] == "fallback" and saved["refinement_history"][-1]["hits"] == 800


def test_zero_hits_after_refinements_trigger_the_fallback(tmp_path, monkeypatch):
    replies = [{"pubmed": BAD, "openalex": "x"}] * 3
    out, saved = run(tmp_path, monkeypatch, replies,
                     lambda q, a, b: 0 if "Conference" in q or "Articles" in q else 900)
    assert saved["fallback_used"] is True and out["queries"]["pubmed"].startswith("Medical[tiab]")
    assert saved["model_pubmed_query"].count('"') > 0                      # normalised model query is kept for the record


def test_a_repairable_query_is_kept_without_fallback(tmp_path, monkeypatch):
    good = "('Deep learning'[MeSH Terms] OR 'deep learning'[Title/Abstract]) AND ('Diagnostic Imaging'[MeSH Terms])"
    out, saved = run(tmp_path, monkeypatch, [{"pubmed": good, "openalex": "deep learning imaging"}], lambda q, a, b: 5000)
    assert saved["fallback_used"] is False and '"Deep learning"[MeSH Terms]' in out["queries"]["pubmed"]
