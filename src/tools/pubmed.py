import os
import time
import xml.etree.ElementTree as ET
import requests

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def _params(extra):
    p = {"tool": "agentic_review", "email": os.getenv("CONTACT_EMAIL", "")}
    if os.getenv("NCBI_API_KEY"):
        p["api_key"] = os.getenv("NCBI_API_KEY")
    p.update(extra)
    return p


def _esearch_params(query, retmax, year_from, year_to):
    es = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"}
    if year_from or year_to:
        es.update({"datetype": "pdat", "mindate": year_from or 1900, "maxdate": year_to or 3000})
    return es


def count(query: str, year_from=None, year_to=None) -> int:
    """Number of PubMed hits for a query (used to refine the search strategy)."""
    r = requests.get(BASE + "esearch.fcgi",
                     params=_params(_esearch_params(query, 0, year_from, year_to)), timeout=60)
    r.raise_for_status()
    return int(r.json()["esearchresult"]["count"])


def search(query: str, max_results: int = 50, year_from=None, year_to=None):
    r = requests.get(BASE + "esearch.fcgi",
                     params=_params(_esearch_params(query, max_results, year_from, year_to)), timeout=60)
    r.raise_for_status()
    ids = r.json()["esearchresult"]["idlist"]
    if not ids:
        return []
    time.sleep(0.4)
    r = requests.post(BASE + "efetch.fcgi", data=_params(
        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}), timeout=120)
    r.raise_for_status()
    return parse_pubmed_xml(r.text)


def parse_pubmed_xml(xml_text: str):
    out = []
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//Abstract/AbstractText"))
        year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
        doi = ""
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""
        authors = [f"{a.findtext('ForeName') or ''} {a.findtext('LastName') or ''}".strip()
                   for a in art.findall(".//AuthorList/Author")][:6]
        out.append({"id": f"pmid:{pmid}", "source": "pubmed", "doi": doi, "title": title,
                    "abstract": abstract, "year": int(year) if str(year).isdigit() else None,
                    "authors": authors, "venue": art.findtext(".//Journal/Title")})
    return out
