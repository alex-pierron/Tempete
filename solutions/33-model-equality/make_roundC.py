"""Build tour C round files (rounds 10+) from the server budget table: spend every answer left, 8 per round per endpoint.

Usage: make_roundC.py <budget.txt> <first_round> <endpoints comma list in priority order> <out_dir>
Long high-entropy prompts (256 tokens) for the base-checkpoint likelihood relecture; each prompt id is new
(no repeat), so the watermark copies do not replay identical text.
"""
import json, sys, datetime, re
TOPICS = ["a lost umbrella", "the first snow of winter", "a robot learning to paint", "an old map found in an attic",
 "a night train across the desert", "a beekeeper and her bees", "a chess game in a park", "a city without electricity for a day",
 "a letter that arrives fifty years late", "a submarine crew hearing music", "a market at dawn", "the last bookshop in town",
 "a violin made from driftwood", "a pilot lost over the ocean", "a garden on a rooftop", "a dog that waits at a station",
 "an astronaut returning home", "a festival of lanterns", "a mountain rescue", "a forgotten language",
 "a baker who never sleeps", "a storm over a fishing village", "a museum at midnight", "a child building a kite",
 "a bridge between two rival towns", "a clock that runs backwards", "a desert oasis", "a whale song recorded by a scientist",
 "an island with one inhabitant", "a tram driver's last day", "a glassblower's apprentice", "a comet seen by shepherds",
 "a cook on an icebreaker", "a radio station at the end of the world", "a librarian and a mysterious borrower", "a carnival leaving town",
 "a tailor who sews memories", "a monastery in the clouds", "a race between two sailing boats", "a photograph that changes",
 "a village that votes on the weather", "a translator at a peace conference", "a lighthouse turned into a school", "a family recipe lost and found",
 "a mapmaker who invents a country", "a long-distance runner in the rain", "a puppet theatre in a war", "a scientist who talks to plants"]
FORMS = ["Write a story of about 300 words about {}.", "Write a 300-word diary entry about {}.",
         "Write a 300-word letter describing {}.", "Describe, in about 300 words, {}."]
def main():
    budget_txt, first_round, eps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(','), sys.argv[4]
    left = {}
    for line in open(budget_txt):
        m = re.match(r'^(ep-\d\d)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if m: left[m.group(1)] = int(m.group(4))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    prompts = [FORMS[i % len(FORMS)].format(t) for i, t in enumerate(TOPICS)]
    want = {e: left[e] for e in eps if left.get(e, 0) > 0}
    rnd, k, total = first_round, 0, 0
    while any(v > 0 for v in want.values()):
        entries = []
        for j in range(8):
            if k + j >= len(prompts): break
            targets = [e for e in eps if want.get(e, 0) > j]
            if targets: entries.append({"id": f"WC-long-{k + j:02d}", "prompt": prompts[k + j], "repeats": 1, "first_repeat": 0, "endpoints": targets})
        if not entries: break
        for e in want: want[e] = max(0, want[e] - 8)
        n = sum(len(x['endpoints']) for x in entries); total += n
        json.dump({"plan": "lucas-met", "round": rnd, "purpose": f"Tour C (spend 100 % of the budget) round {rnd}: long answers for the base-checkpoint likelihood relecture.",
                   "frozen_at_utc": now, "max_tokens": 256, "prompts": entries}, open(f"{out}/round{rnd}_prompts.json", "w"), indent=1)
        print(f"round {rnd}: {n} answers"); rnd += 1; k += 8
    print("total", total)
main()
