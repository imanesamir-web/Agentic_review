import json
import os

import yaml
from src.llm import chat_json
from src.audit import log, save_json

SYSTEM = ("You are an expert systematic-review methodologist (PRISMA 2020). "
          "Fill the review protocol for the research question. Keep every key of the template. "
          "Be specific: PICO elements, 4-6 inclusion and 4-6 exclusion criteria, study designs.")


def planner(state):
    with open("config/protocol_template.yaml", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    user = f"Research question: {state['question']}\n\nTemplate (JSON/YAML):\n{yaml.dump(template)}"
    proto = chat_json(SYSTEM, user)
    # keep the user-controlled settings from the template
    for key in ("databases", "date_range", "languages", "max_records_per_database",
                "screening", "extraction_fields", "human_in_the_loop"):
        proto[key] = template[key]
    proto["research_question"] = state["question"]
    save_json(state["run_dir"], "protocol.json", proto)
    log(state["run_dir"], "planner", "protocol_generated", proto, prompt=SYSTEM + user)
    if os.getenv("HUMAN_APPROVAL", "0") == "1":
        proto = approve_protocol(state["run_dir"], proto)
    return {"protocol": proto}


def approve_protocol(run_dir, proto):
    """Human-in-the-loop checkpoint: you may edit protocol.json (criteria, PICO...) before anything is searched."""
    path = os.path.join(run_dir, "protocol.json")
    print("\n[human-in-the-loop] Protocol generated. Review it in:\n  " + path, flush=True)
    print("  Inclusion: " + " | ".join(proto.get("inclusion_criteria", [])), flush=True)
    print("  Exclusion: " + " | ".join(proto.get("exclusion_criteria", [])), flush=True)
    input("Edit the file if needed (keep it valid JSON) and save it, then press Enter to continue... ")
    try:
        with open(path, encoding="utf-8") as f:
            new = json.load(f)
    except Exception as e:
        print(f"[human-in-the-loop] could not read the edited file ({e}); keeping the generated protocol.", flush=True)
        new = proto
    log(run_dir, "planner", "protocol_approved", {"edited": new != proto})
    return new
