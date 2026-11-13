"""Evaluation commands (screening accuracy, extraction edits, time savings).
  py evaluate.py screening-sheet runs\\run_X [N]   make the BLIND labelling sheet (default 30 records)
  py evaluate.py screening-score runs\\run_X       score the agents against your filled-in sheet
  py evaluate.py extraction runs\\run_X            how much of the LLM extraction you had to correct
  py evaluate.py time runs\\run_X                  automated time vs an editable manual-effort estimate
  py evaluate.py critic-canary runs\\run_X        plant known errors in the narrative: does the critic catch them?
  py evaluate.py critic-summary runs\\run_X       what is behind the critic's counts (by check, with examples)
  py evaluate.py critic-review-sheet runs\\run_X [N]   sample of the critic's findings for YOU to judge (default 20)
  py evaluate.py critic-review-score runs\\run_X       precision of each kind of finding from your verdicts
  py evaluate.py screening-disagreements runs\\run_X  where the pipeline and your labels differ, with the pipeline's reason
  py evaluate.py extraction-sheet runs\\run_X [N]     verification sample: mark the extracted values correct/partial/wrong
  py evaluate.py extraction-score runs\\run_X         accuracy of the extraction from your verdicts"""
import sys

from src import evaluation as ev

if len(sys.argv) < 3:
    sys.exit(__doc__)
cmd, run_dir = sys.argv[1], sys.argv[2]
if cmd == "screening-sheet":
    sheet, key = ev.make_screening_sheet(run_dir, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    print(f"Sheet: {sheet}\nCriteria: {run_dir}\\eval\\criteria.txt\n"
          f"Fill the 'human_decision' column (include / exclude / maybe) WITHOUT looking at the pipeline's results,\n"
          f"save as CSV, then run: py evaluate.py screening-score {run_dir}")
elif cmd == "screening-score":
    print(ev.format_screening(ev.score_screening(run_dir)))
elif cmd == "extraction":
    print(ev.format_extraction(ev.extraction_changes(run_dir)))
elif cmd == "critic-canary":
    print(ev.format_canary(ev.critic_canary(run_dir)))
elif cmd == "critic-review-sheet":
    sheet, sizes = ev.critic_review_sheet(run_dir, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    print(f"Sheet: {sheet}\nFindings available per type: {sizes}\nFill 'human_verdict' (valid / invalid / unsure): "
          f"is the critic's finding a REAL problem in the sentence, given the evidence? Then run: "
          f"py evaluate.py critic-review-score {run_dir}")
elif cmd == "critic-review-score":
    print(ev.format_critic_review(ev.score_critic_review(run_dir)))
elif cmd == "screening-disagreements":
    print(ev.format_disagreements(ev.screening_disagreements(run_dir)))
elif cmd == "extraction-sheet":
    sheet, info = ev.extraction_sheet(run_dir, int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    print(f"Sheet: {sheet}\n{info}\nFor each row mark 'verdict': correct / partial / wrong / unverifiable (not checkable "
          f"from the abstract: open the paper through its DOI or mark unverifiable). Then run: "
          f"py evaluate.py extraction-score {run_dir}")
elif cmd == "extraction-score":
    print(ev.format_extraction_verify(ev.score_extraction_sheet(run_dir)))
elif cmd == "critic-summary":
    print(ev.format_critic_summary(ev.critic_summary(run_dir)))
elif cmd == "time":
    print(ev.format_time(ev.time_report(run_dir)))
else:
    sys.exit(__doc__)
