# utils.py
import json, time, os
from pathlib import Path

def dump_state(state: dict, base_dir: str = "data/storage"):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    fname = f"state_{state['ciclo_id']:08d}.json"
    with open(os.path.join(base_dir, fname), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
