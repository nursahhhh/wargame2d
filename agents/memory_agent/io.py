import json
from pathlib import Path
from typing import List, Dict


def load_all_reflections(reflections_dir: str) -> List[Dict]:
    reflections = []
    for path in Path(reflections_dir).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "failure_analysis" in data:
            reflections.append(data["failure_analysis"])
    return reflections


def write_experience_guidance(output_path: str, guidance: Dict):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(guidance, indent=2),
        encoding="utf-8"
    )
