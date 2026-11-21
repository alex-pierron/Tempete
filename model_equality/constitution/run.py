"""Send a question file of the constitution to the reference endpoints ref-01 ... ref-09 only.

    python constitution/run.py constitution/prompts_v0_AB.jsonl [repeats=3] [ref-01,ref-02]

Answers go to constitution/answers_<file stem>.jsonl (official format, safe restart). Never queries ep-*.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
_env = HERE.parents[1] / ".env"
if not os.environ.get("HACKATHON_API_KEY") and _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        if _line.startswith("HACKATHON_API_KEY=") and _line.split("=", 1)[1].strip():
            os.environ["HACKATHON_API_KEY"] = _line.split("=", 1)[1].strip()
import query_reference as qr  # noqa: E402


def main() -> None:
    prompts_path = Path(sys.argv[1])
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with prompts_path.open(encoding="utf-8") as f:
        prompts = [(r["id"], r["prompt"]) for r in (json.loads(line) for line in f if line.strip())]
    endpoints = [ep for ep in qr.get_budget(qr.make_session()) if ep.startswith("ref-")]
    if len(sys.argv) > 3:  # optional comma-separated subset, e.g. ref-01,ref-02
        endpoints = [ep for ep in sys.argv[3].split(",") if ep in endpoints]
    assert endpoints and all(ep.startswith("ref-") for ep in endpoints)
    qr.query(prompts, output=HERE / f"answers_{prompts_path.stem}.jsonl", endpoints=endpoints, repeats=repeats)


if __name__ == "__main__":
    main()
