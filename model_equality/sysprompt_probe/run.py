"""Send the system-prompt probes to the reference endpoints ref-01 ... ref-09 (unlimited, never scored).

Reuses query() from the official query_reference.py, so the answer format, the retries and the safe restart are the
official ones. Answers go to their own file, so the team's reference_answers.jsonl is not touched.

    export HACKATHON_API_KEY=...        # never commit the key
    cd model_equality
    python sysprompt_probe/run.py       # 9 prompts x 9 endpoints x 3 repeats = 243 requests on ref-* only

Never sends anything to ep-*: the endpoint list is filtered to ref-* before querying.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# Read HACKATHON_API_KEY from the repository's .env (git-ignored) if it is not already in the environment.
_env = HERE.parents[1] / ".env"
if not os.environ.get("HACKATHON_API_KEY") and _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        if _line.startswith("HACKATHON_API_KEY=") and _line.split("=", 1)[1].strip():
            os.environ["HACKATHON_API_KEY"] = _line.split("=", 1)[1].strip()
import query_reference as qr  # noqa: E402  (official script, one folder up)

PROMPTS = HERE / "prompts.jsonl"
OUTPUT = HERE / "answers.jsonl"
REPEATS = 3


def main() -> None:
    if qr.API_KEY == "YOUR_API_KEY_HERE" or not qr.API_KEY.strip():
        raise SystemExit("Set HACKATHON_API_KEY first.")
    with PROMPTS.open(encoding="utf-8") as f:
        prompts = [(r["id"], r["prompt"]) for r in (json.loads(line) for line in f if line.strip())]
    endpoints = [ep for ep in qr.get_budget(qr.make_session()) if ep.startswith("ref-")]
    assert endpoints and all(ep.startswith("ref-") for ep in endpoints)
    print("Reference endpoints:", ", ".join(endpoints))
    qr.query(prompts, output=OUTPUT, endpoints=endpoints, repeats=REPEATS)


if __name__ == "__main__":
    main()
