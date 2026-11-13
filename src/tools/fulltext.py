"""Open-access full-text retrieval. Routes, in order:
 1. direct PDF link (OpenAlex)  2. Europe PMC full-text XML (by PMID or DOI)  3. Unpaywall PDF  4. OA landing URL.
PDFs are parsed with pypdf (pure Python). Text is cached on disk for reproducibility."""
import io
import logging
import os
import re
import xml.etree.ElementTree as ET

import requests

logging.getLogger("pypdf").setLevel(logging.ERROR)  # hides harmless font/encoding warnings

MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 30
MIN_CHARS = 2000
UA = {"User-Agent": "agentic-review/0.1"}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def safe_name(rec_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", rec_id)[-80:]


def unpaywall_pdf(doi: str) -> str:
    mail = os.getenv("CONTACT_EMAIL", "")
    if not doi or not mail:
        return ""
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": mail},
                         headers=UA, timeout=15)
        if r.status_code != 200:
            return ""
        loc = r.json().get("best_oa_location") or {}
        return loc.get("url_for_pdf") or ""
    except Exception:
        return ""


def pdf_bytes_to_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages[:MAX_PAGES]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return re.sub(r"\s+", " ", " ".join(pages)).strip()


def jats_to_text(xml) -> str:
    """Body text (section titles + paragraphs) of a JATS full-text XML (Europe PMC / PMC)."""
    root = ET.fromstring(xml)
    body = root.find(".//body")
    if body is None:
        return ""
    parts = []
    for el in body.iter():
        if el.tag in ("title", "p"):
            t = " ".join("".join(el.itertext()).split())
            if t:
                parts.append(t)
    return " ".join(parts)


def epmc_fulltext(rec: dict):
    """Returns (text, source_url) on success or (None, reason)."""
    rid = rec.get("id", "")
    pmid = rid.split(":", 1)[1] if rid.startswith("pmid:") else ""
    doi = rec.get("doi", "")
    if pmid:
        q = f"EXT_ID:{pmid} AND SRC:MED"
    elif doi:
        q = f'DOI:"{doi}"'
    else:
        return None, "epmc:no_identifier"
    try:
        r = requests.get(f"{EPMC}/search", params={"query": q, "format": "json",
                                                   "resultType": "lite", "pageSize": 1},
                         headers=UA, timeout=20)
        hits = r.json()["resultList"]["result"]
        if not hits:
            return None, "epmc:not_found"
        h = hits[0]
        if not h.get("pmcid") or h.get("isOpenAccess") != "Y":
            return None, "epmc:not_open_access"
        url = f"{EPMC}/{h['pmcid']}/fullTextXML"
        x = requests.get(url, headers=UA, timeout=30)
        if x.status_code != 200:
            return None, f"epmc:http_{x.status_code}"
        text = jats_to_text(x.content)
    except Exception as e:
        return None, f"epmc:{type(e).__name__}"
    if len(text) < MIN_CHARS:
        return None, "epmc:too_short"
    return text, url


def _download(url: str):
    r = requests.get(url, headers=UA, timeout=(10, 30), stream=True)
    if r.status_code != 200:
        return None
    data = b""
    for chunk in r.iter_content(65536):
        data += chunk
        if len(data) > MAX_BYTES:
            return None
    return data if data[:5] == b"%PDF-" else None


def _pdf_text(url: str):
    """Returns (text, url) on success or (None, reason)."""
    try:
        data = _download(url)
    except Exception as e:
        return None, f"pdf:download_error_{type(e).__name__}"
    if data is None:
        return None, "pdf:not_a_pdf"
    try:
        text = pdf_bytes_to_text(data)
    except Exception as e:
        return None, f"pdf:parse_error_{type(e).__name__}"
    if len(text) < MIN_CHARS:
        return None, "pdf:no_text_layer"
    return text, url


def fetch_fulltext(rec: dict, cache_dir: str):
    """Returns (text or None, info). Text is cached under cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, safe_name(rec["id"]) + ".txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read(), {"source_url": "cache"}
    reasons = []
    attempts = []
    if rec.get("oa_pdf"):
        attempts.append(lambda: _pdf_text(rec["oa_pdf"]))
    attempts.append(lambda: epmc_fulltext(rec))
    attempts.append(lambda: (lambda u: _pdf_text(u) if u else (None, "unpaywall:no_pdf_url"))(
        unpaywall_pdf(rec.get("doi", ""))))
    if rec.get("oa_url") and rec.get("oa_url") != rec.get("oa_pdf"):
        attempts.append(lambda: _pdf_text(rec["oa_url"]))
    for attempt in attempts:
        text, src = attempt()
        if text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return text, {"source_url": src}
        reasons.append(src)
    return None, {"reason": "; ".join(reasons)}
