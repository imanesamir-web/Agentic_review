from typing import TypedDict, List, Dict, Any


class ReviewState(TypedDict, total=False):
    run_id: str
    run_dir: str
    question: str
    protocol: Dict[str, Any]
    queries: Dict[str, str]
    records: List[Dict[str, Any]]      # deduplicated records
    screened: List[Dict[str, Any]]     # title/abstract screening rows
    included: List[Dict[str, Any]]     # passed title/abstract screening
    extracted: List[Dict[str, Any]]    # eligibility + extraction + risk of bias per record
    final_included: List[Dict[str, Any]]
    synthesis: Dict[str, Any]          # studies, groups, tables, narrative blocks
    critic: Dict[str, Any]             # QA report (issues, counts, history, verdict)
    critic_feedback: Dict[str, Any]    # passage key -> problems to fix on the next synthesis pass
    revision_count: int
    best_pass: Dict[str, Any]          # best synthesis version seen so far (fewest high-severity findings)
    report_path: str
    prisma: Dict[str, Any]
    needs_revision: bool
