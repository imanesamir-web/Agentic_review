import re


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def deduplicate(records):
    """Dedup by DOI, then by normalized title. Returns (unique, removed) with the reason."""
    seen_doi, seen_title, unique, removed = {}, {}, [], []
    for r in records:
        doi = (r.get("doi") or "").lower().strip()
        nt = norm_title(r.get("title"))
        if doi and doi in seen_doi:
            removed.append({"id": r["id"], "duplicate_of": seen_doi[doi], "reason": "same DOI"})
            continue
        if nt and nt in seen_title:
            removed.append({"id": r["id"], "duplicate_of": seen_title[nt], "reason": "same title"})
            continue
        if doi:
            seen_doi[doi] = r["id"]
        if nt:
            seen_title[nt] = r["id"]
        unique.append(r)
    return unique, removed
