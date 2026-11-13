# Agentic AI pipeline for systematic reviews (PRISMA 2020)

A multi-agent system that carries out a systematic literature review end to end from a research question:
protocol, search strategy, retrieval and deduplication, dual title/abstract screening, full-text eligibility,
structured extraction with risk-of-bias appraisal, narrative synthesis, an automated quality critic, and a final
report with a PRISMA flow diagram and BibTeX. Every decision is logged, every stage can be re-run on its own, and a
human can inspect and correct the intermediate files.

Orchestration: **LangGraph**. Retrieval over papers: **BM25 over full-text chunks**. Sources: **PubMed, OpenAlex,
Europe PMC, Unpaywall**. Models: any OpenAI-compatible server (local **Ollama** or **OpenRouter**).

## Pipeline

```mermaid
flowchart LR
  Q[Research question] --> P[Planner: protocol]
  P --> S[Search strategy: queries + hit-count refinement]
  S --> R[Retrieval + dedup]
  R --> SC[Screening: 2 reviewers + arbiter]
  SC --> E[Extraction + risk of bias]
  E --> SY[Synthesis: groups, tables, narrative]
  SY --> C{Critic}
  C -- high-severity findings, better version --> SY
  C --> RP[Report: md/html, PRISMA svg, bib]
```

| Stage | What it does | Main output (in `runs/<run_id>/`) |
|---|---|---|
| Planner | Fills the PRISMA protocol template (PICO, inclusion/exclusion criteria) | `protocol.json` |
| Search strategy | Writes a PubMed query and an OpenAlex query; lints the query, checks the PubMed hit count, refines it if too narrow | `queries.json` |
| Retrieval | Queries both databases, deduplicates by DOI then title | `records.json`, `prisma_counts.json` |
| Screening | Two LLM reviewers screen titles/abstracts (5 records per call); an arbiter resolves disagreements; reasons are coded against numbered criteria (E1, E2...); Cohen's kappa is computed | `screening.json`, `included_records.json` |
| Extraction | Finds open-access full text (OpenAlex, Europe PMC, Unpaywall), retrieves relevant chunks with BM25, re-checks eligibility, extracts fields with their source, rates 5 risk-of-bias items (overall rating computed by code) | `extraction.json`, `evidence_table.csv`, `fulltext/` |
| Synthesis | Groups studies (by task or imaging modality), builds evidence and risk-of-bias tables, checks metric comparability, writes a cited narrative | `synthesis.json`, `synthesis_table.csv`, `group_assignments.json` |
| Critic | Checks the narrative against the extracted data and the protocol; sends flagged passages back for rewriting; keeps the best version | `critic_report.json` |
| Report | Assembles methods, results, limitations, QA and reproducibility sections | `final_report.md/.html`, `prisma_flow.svg`, `references.bib` |

## Methodological choices

- **No pooling of performance metrics.** A descriptive range is shown only when at least three studies report the
  same metric on the same *named* evaluation dataset; otherwise results are given study by study.
- **Transparency.** Every agent decision goes to `decisions.jsonl` (stage, model, prompt hash, rationale). PRISMA counts
  are stored per stage. Studies assessed on the abstract only are flagged, and the report states how many.
- **Critic.** Deterministic checks: citations exist (and belong to the group), every number is found in the cited
  studies' extracted data, no pooling/averaging wording across studies, PRISMA arithmetic, group coverage, date range.
  Optional LLM check of each passage against its cited studies. A rewrite is accepted only if it has fewer high-severity
  findings than the best version so far. Unresolved findings stay in the report, marked `⚠[unverified]`.
- **Human oversight.** Set `HUMAN_APPROVAL=1` to pause after the protocol and after the grouping so you can edit
  `protocol.json` / `group_assignments.json`. Rows of `extraction.json` that you correct by hand can be protected with
  `"curated": true`. Labelling sheets (below) measure the agents against your own judgement.

## Installation

Python 3.10 or newer (developed on 3.14, Windows 11). No GPU is needed when you use an API; a GPU speeds up a local model.

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then edit .env
py -m pytest tests -q       # offline tests, no network or model needed
```

Notes
- The code deliberately uses pure-Python dependencies (`requests`, `pypdf`, `rank-bm25`, `markdown`): on machines with an
  application-control policy, compiled packages (e.g. `jiter`, `xxhash`) can be blocked.
- Local model (recommended when API quotas run out): install [Ollama](https://ollama.com), run
  `ollama pull llama3.1:8b`, set `OLLAMA_CONTEXT_LENGTH=8192`, restart Ollama, then `py check_llm.py`.

## Configuration (`.env`)

See `.env.example` for the full annotated list. The most important settings:

| Variable | Meaning | Default |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Backend and model (Ollama or OpenRouter) | OpenRouter |
| `LLM_JSON_MODE` | Ask the server for JSON-only output (use `1` with Ollama) | `0` |
| `LLM_TIMEOUT` | Seconds to wait for one model answer | `300` |
| `LLM_FALLBACK_MODELS` | Comma-separated models tried when one is rate-limited | empty |
| `CONTACT_EMAIL` | Real e-mail for OpenAlex, NCBI and Unpaywall | - |
| `SCREEN_MODEL_B` | Different model for screening reviewer B | same model |
| `HUMAN_APPROVAL` | Pause for human edits (protocol, grouping) | `0` |
| `SYNTH_GROUP_BY` | `auto`, `modality` or `task` | `auto` |
| `CRITIC_CITATION_SCOPE` | `paragraph`, `inherit` or `sentence` | `paragraph` |
| `CRITIC_SEMANTIC` / `CRITIC_MODEL` | LLM check of the narrative / model for it | `0` / main model |
| `CRITIC_LLM_REWRITES` | `0`: LLM findings are advisory, do not trigger rewrites | `1` |
| `MAX_REVISIONS` | Rewrite passes requested by the critic | `2` |
| `SCREEN_LIMIT`, `EXTRACT_LIMIT` | Process only the first N records (quick tests) | all |

The review protocol template (databases, date range, records per database, extraction fields) is `config/protocol_template.yaml`.

## Usage

Full run, from a question to the report:

```
py run.py "Medical image classification using deep learning"
```

Single stages on an existing run folder (results are cached; nothing earlier is repeated):

```
py run_extraction.py runs\run_X            # full text + extraction + risk of bias (retries only failed records)
py run_synthesis.py  runs\run_X            # synthesis -> critic -> report
py run_synthesis.py  runs\run_X --reuse    # keep the narrative, only critic + report
py check_llm.py                            # test the configured model
py diag_fulltext.py runs\run_X 10          # why was full text (not) found?
```

Typical workflow with human curation: run the pipeline, open `evidence_table.csv` / `extraction.json`, correct values,
add `"curated": true` to the corrected rows, edit `group_assignments.json` if a group looks wrong, then
`py run_synthesis.py runs\run_X`. Backups of overwritten files are kept in `runs\run_X\backups\`.

### Evaluation commands

```
py evaluate.py screening-sheet runs\run_X 30   # blind sheet: label include/exclude yourself
py evaluate.py screening-score runs\run_X      # sensitivity, specificity, precision, F1, kappa vs your labels
py evaluate.py extraction runs\run_X           # how many extracted fields you had to correct
py evaluate.py time runs\run_X                 # automated time vs an editable manual-effort estimate
py evaluate.py critic-canary runs\run_X        # plant known errors: does the critic catch them?
py evaluate.py critic-summary runs\run_X       # what the critic found, by check, with examples
py evaluate.py critic-review-sheet runs\run_X  # judge a sample of the critic's findings (precision)
```

## Project structure

```
run.py  run_extraction.py  run_synthesis.py  evaluate.py  check_llm.py  diag_fulltext.py
config/protocol_template.yaml
src/
  graph.py            LangGraph workflow (+ critic/revision loop)
  state.py  llm.py  audit.py  timing.py  fileutil.py  metrics.py  evaluation.py  synthesis_utils.py
  agents/             planner, search_strategy, retrieval, screening, extraction, synthesis, critic, report
  tools/              pubmed, openalex, dedup, fulltext (OpenAlex / Europe PMC / Unpaywall + pypdf)
  rag/                chunks.py (BM25 over full-text chunks), index.py (dense+sparse index, optional)
tests/                offline tests (fake models and databases), including a full-graph smoke test
runs/<run_id>/        everything a run produces (ignored by git)
```

## How the project was developed

1. Defined the architecture (planner, search, retrieval, screening, extraction, synthesis, critic) and the PRISMA protocol template.
2. Built the skeleton: shared state, backend-agnostic LLM client, audit trail, OpenAlex/PubMed tools, deduplication.
3. Implemented dual screening with an arbiter, coded exclusion reasons and Cohen's kappa; batched calls for speed.
4. Added full-text retrieval (OpenAlex, Europe PMC, Unpaywall), BM25 chunk retrieval, structured extraction with sources and risk-of-bias rating.
5. Added synthesis (grouping, evidence tables, comparability rule, cited narrative), the critic with a revision loop, and the report generator.
6. Hardened it from real runs: rate-limit fallback and cooldown, resume of failed records, protection of hand-curated rows, backups, locked-file handling.
7. Evaluated the system: screening and extraction against human judgement, time against a manual estimate, and the critic with seeded errors.

## Known limitations

- Small local models cite once per paragraph, follow formatting less reliably and make a noisy LLM critic; larger models improve this.
- The search is not exhaustive (a capped number of records per database; PubMed and OpenAlex only).
- Risk-of-bias ratings for abstract-only records reflect what the abstract reports, not how the study was conducted.
- Agreement between agents (kappa) measures consistency, not correctness; use the evaluation commands for accuracy.
- The `evaluate.py time` manual-effort figures are editable assumptions, not measured values.
