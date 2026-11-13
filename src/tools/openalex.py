import os
import requests

URL = "https://api.openalex.org/works"


def rebuild_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def search(query: str, max_results: int = 50, year_from=None, year_to=None):
    flt = ["has_abstract:true"]
    if year_from:
        flt.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        flt.append(f"to_publication_date:{year_to}-12-31")
    params = {"search": query, "per-page": min(max_results, 200), "filter": ",".join(flt)}
    mail = os.getenv("CONTACT_EMAIL")
    if mail:
        params["mailto"] = mail
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    out = []
    for w in r.json().get("results", [])[:max_results]:
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        best = w.get("best_oa_location") or {}
        out.append({
            "id": w.get("id"), "source": "openalex", "doi": doi,
            "title": w.get("title") or "",
            "abstract": rebuild_abstract(w.get("abstract_inverted_index")),
            "year": w.get("publication_year"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])][:6],
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "oa_pdf": best.get("pdf_url") or "",
            "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
        })
    return out
