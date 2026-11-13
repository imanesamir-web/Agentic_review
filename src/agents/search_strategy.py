"""Search Strategy agent: drafts the query, lints it, checks the PubMed hit count, and refines it."""
import re

from src.llm import chat_json
from src.audit import log, save_json
from src.tools import pubmed

MAX_REFINE = 2

SYSTEM = (
    "You design database search strategies for systematic reviews (PRISMA 2020). "
    'Return JSON {"pubmed": "...", "openalex": "...", "rationale": "..."}.\n'
    "PUBMED rules:\n"
    "- Combine AT MOST 3 concept blocks with AND (condition/domain, intervention/method, optionally task).\n"
    "- Each block = MeSH terms with [MeSH Terms] plus free-text synonyms with [Title/Abstract], joined by OR.\n"
    "- Do NOT add blocks for outcomes, metrics, study design, validation or datasets: they hurt recall.\n"
    "- Only allowed filters: English[Language] and a final exclusion "
    "NOT (Review[Publication Type] OR Editorial[Publication Type] OR Letter[Publication Type] OR Comment[Publication Type]).\n"
    "- Never invent publication types (there is no 'Conference Abstract'). Balanced parentheses and quotes.\n"
    "OPENALEX: a short natural-language keyword query (6-12 key terms, no field tags)."
)


def _query_text(value):
    """Extract query text when an LLM returns a structured JSON field."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("query", "search_query", "search_terms", "text", "value", "strategy"):
            if key in value:
                result = _query_text(value[key])
                if result:
                    return result
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(text for item in value if (text := _query_text(item)))
    return ""


def lint_pubmed(q: str):
    problems = []
    if not isinstance(q, str):
        return ["query must be text"]
    if not q or not q.strip():
        return ["empty query"]
    if q.count("(") != q.count(")"):
        problems.append("unbalanced parentheses")
    if q.count('"') % 2:
        problems.append("unbalanced double quotes")
    if "conference abstract" in q.lower():
        problems.append("'Conference Abstract' is not a PubMed publication type")
    if q.count("[") != q.count("]"):
        problems.append("unbalanced square brackets")
    return problems

STOP = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "using", "use", "by", "from", "at", "as", "is",
        "are", "be", "or", "their", "its", "into", "via", "based", "between", "among", "what", "which", "how", "does",
        "do", "effect", "effects", "review", "study", "studies"}


def normalize_pubmed_query(q: str) -> str:
    """Repairs the mistakes small models make most often: single-quoted phrases (PubMed only understands double
    quotes), typographic quotes, and 'AND NOT' (PubMed wants 'NOT')."""
    q = _query_text(q).replace("\u201c", '"').replace("\u201d", '"')
    q = re.sub(r"'([^'\[\]()]+)'(\s*\[)", r'"\1"\2', q)
    return re.sub(r"\bAND\s+NOT\b", "NOT", q, flags=re.I)


def keyword_pubmed_query(question: str) -> str:
    """Always-valid fallback: the content words of the research question, each searched in title/abstract."""
    seen, words = set(), []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question or ""):
        if w.lower() not in STOP and w.lower() not in seen:
            seen.add(w.lower())
            words.append(w)
    return " AND ".join(f"{w}[tiab]" for w in words[:6]) or "systematic[tiab]"


def search_strategy(state):
    p, run_dir = state["protocol"], state["run_dir"]
    user = (f"Question: {p['research_question']}\nPICO: {p['pico']}\n"
            f"Inclusion: {p['inclusion_criteria']}\nExclusion: {p['exclusion_criteria']}")
    out = chat_json(SYSTEM, user)
    min_hits = max(30, int(p.get("max_records_per_database", 50)))
    yf, yt = p["date_range"]["from"], p["date_range"]["to"]
    history = []
    for attempt in range(MAX_REFINE + 1):
        q = normalize_pubmed_query(out.get("pubmed", ""))
        problems, n = lint_pubmed(q), None
        if not problems:
            try:
                n = pubmed.count(q, yf, yt)
                if n < min_hits:
                    problems.append(f"too narrow: only {n} PubMed hits. Remove the least essential "
                                    "concept block or add synonyms to the remaining blocks")
            except Exception as e:
                print(f"[search] could not check hit count: {e}", flush=True)
        history.append({"attempt": attempt, "pubmed": q, "hits": n, "problems": problems})
        print(f"[search] attempt {attempt}: hits={n} problems={problems}", flush=True)
        if not problems or attempt == MAX_REFINE:
            break
        out = chat_json(SYSTEM, user + f"\n\nYour previous PubMed query:\n{q}\n"
                                       f"Problems found: {problems}\nReturn a corrected full JSON.")
    final_q = q
    fallback_used = False
    last = history[-1]
    lint_failed = any(("unbalanced" in x or "empty" in x or "not a PubMed" in x) for x in last["problems"])
    if lint_failed or last["hits"] == 0:       # never send an invalid or empty-result query to PubMed unchecked
        fb = keyword_pubmed_query(p["research_question"])
        try:
            fb_hits = pubmed.count(fb, yf, yt)
        except Exception as e:
            fb_hits = None
            print(f"[search] could not check the fallback hit count: {e}", flush=True)
        history.append({"attempt": "fallback", "pubmed": fb, "hits": fb_hits, "problems": []})
        print(f"[search] model query unusable: fallback keyword query used (hits={fb_hits}): {fb}", flush=True)
        final_q, fallback_used = fb, True
    openalex_query = _query_text(out.get("openalex")) or " ".join(
        re.findall(r"[A-Za-z][A-Za-z0-9\-]+", p["research_question"]))
    out = {**out, "openalex": openalex_query}
    queries = {"pubmed": final_q, "openalex": openalex_query}
    save_json(run_dir, "queries.json", {**out, "pubmed": final_q, "model_pubmed_query": q,
                                        "fallback_used": fallback_used, "refinement_history": history})
    log(run_dir, "search_strategy", "queries_generated", {**out, "pubmed": final_q, "fallback_used": fallback_used,
                                                         "history": history}, prompt=SYSTEM + user)
    return {"queries": queries}
