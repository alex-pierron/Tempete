"""Fresh prompt waves of research team D (`r-met-d`), NOT the official bank.

Team D looks for what betrays fp8 quantization (T1) and weak LoRA (T2, strength 0.3) on the
unlimited `ref-*` endpoints. The prompts are *new* (ids `d-...`), so that the statistics chosen on
the answers collected by teams A and B (discovery) can be validated on answers nobody looked at
before (validation).

Waves:
    d1  short-answer prompts (arithmetic, numbers, yes/no, choices, completions): format-mode
        statistics, the cheapest requests of the server (about 10 s each).
    d2  long structured outputs (counting, copying, random picks, tables): many decisions per
        answer, deterministic chains whose first divergence arrives late.

    d6  echo prompts for fp8 (24 fresh instructions of 6 templates, seed `PROMPT_SEED + 5`). Statistic and
        thresholds written BEFORE any d6 answer: primary = AUC of `log_chars` (fp8 against the S group) at
        N = 20, success >= 0.65, failure <= 0.55; secondary = `echo_exact`. Discovery (post hoc, 29 cells)
        had given 0.72 at N = 10 on the older echo prompts.
    d7  replication of d4: 30 NEW arithmetic one-liners (seed `PROMPT_SEED + 6`), 2 repeats each. Statistic
        `ends_period`; success = AUC >= 0.65 at N = 20 for LoRA 0.3 and band rule (period rate >= 0.6)
        true-positive rate >= 0.7 with false-positive rate <= 0.05 at N = 20; fixed before any answer.
    h1  HOLD-OUT test set (priority 1, asked by the human team): 60 entirely new arithmetic one-liners
        (seed `PROMPT_SEED + 7`, disjoint from d1, d4 and d7), operators interleaved so that every
        prefix of 25 or 50 prompts is balanced, collected 2 repeats x 9 ref. Written and committed
        BEFORE any answer; analysed only by r-met-c with a chain frozen in advance.
    d8  IRIS-type sampling distribution (suggested by the literature through the director): two prompts
        that make the model produce RANDOM symbols, 30 repeats each, on the 5 S endpoints, fp8 and LoRA 0.3.
        Statistic and thresholds written BEFORE any d8 answer: for each prompt the symbol histogram of an
        endpoint is compared with the plain S endpoint (ref-01) by a chi-square distance and a permutation
        test over the responses (2 000 label permutations, response = unit). fp8 is called SEPARATED when its
        permutation p-value is <= 0.01 on at least one prompt AND every disguise (system prompt, persona,
        watermark, temperature 1.0) has a p-value > 0.01 on that prompt; otherwise NOT separated. The
        number of responses needed is then read from the observed distance (estimate).
    d9  IDENTITY flag for disguises (idea of Lucas, via the director): the base model of the ref
        answers "SmolLM / Hugging Face" when asked who it is, except behind a system prompt or a persona.
        3 frozen wordings x 10 repeats x 9 ref. Criterion written BEFORE any d9 answer: a reply is MARKED
        "disguised" when it mentions none of `IDENTITY_BASE_NAMES` (case-insensitive). An endpoint is
        FLAGGED disguised when more than half of its replies are marked. Expected and pre-registered:
        system prompt and persona flagged with a marking rate >= 0.8; plain, watermark, fp8, LoRA 0.3,
        LoRA 0.7 and LoRA 1.5 not flagged with a marking rate <= 0.2. The number of replies per endpoint
        needed is the smallest n whose resampled true-positive rate on system prompt and persona is >= 0.9
        and false-positive rate on the other endpoints <= 0.05.
    d10 / d11  LUCAS' constitution (report of 21/11/2026, section 3.4), rebuilt from its description and
        frozen BEFORE any answer. d10 = block B (families, 4 questions, 5 draws x 9 ref). d11 = block C
        (models, fp8 first, 36 short-answer questions). Rules written before the answers: stage 1 = draws
        0-4 of ref-01 and ref-02; the answer of a question is reduced to a key (digits correct, number,
        letter, first word); a question is RETAINED when the total-variation distance between the key
        distributions of ref-01 and ref-02 is >= 0.6. Stage 2 (confirmation, retained questions only,
        draws 5-9 of both endpoints, independent of the selection): CONFIRMED when the distance is again
        >= 0.6. Stage 3 (disguises, confirmed questions, draws 0-4 on ref-03, ref-05, ref-06, ref-08 and
        ref-09): a question RESISTS a disguise when its distance to ref-01 is <= 0.4.
    d5  fact batteries: 4 variants of 12 trivia questions (seed `PROMPT_SEED + 4`), to replicate the
        pilot in which LoRA 0.3 answered a fact battery with full sentences and more correct items.
    d4  validation of the period-ending signal: 30 fresh arithmetic one-liners (seed
        `PROMPT_SEED + 3`), written down BEFORE any answer is looked at; statistic fixed in advance
        (share of replies ending with a period, endpoint against endpoint).
    d3  batteries: ONE request carries 10 to 20 items (arithmetic, yes/no, random picks, cloze,
        trivia), because the server latency is a queue-dominated fixed cost per request (about
        8 s idle, 100 s or more under load): items per request is what buys statistical power.

Every prompt is generated from the named seed below, so the lists are reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

PROMPT_SEED = 20261121
"""Seed of the generated prompts (arithmetic operands, random word lists)."""
MAX_TOKENS = 256
"""Server cap on generated tokens. The latency of a request is dominated by a fixed floor (about
8 s), so asking for the full cap costs little and keeps the length distribution comparable with
the answers of teams A and B."""
ID_PREFIX = "d"
WAVE_SHORT = "d1"
WAVE_LONG = "d2"
WAVE_BATTERY = "d3"
WAVE_VALIDATION = "d4"
WAVE_FACTS = "d5"
WAVE_ECHO = "d6"
WAVE_REPLICATION = "d7"
WAVE_HOLDOUT = "h1"
WAVE_RANDOM = "d8"
WAVE_IDENTITY = "d9"
WAVE_LUCAS_B = "d10"
WAVE_LUCAS_C = "d11"

ARITHMETIC_PROMPTS_PER_OPERATOR = 3
VALIDATION_PROMPTS_PER_OPERATOR = 10
ARITHMETIC_OPERATORS = ("+", "-", "*")
ARITHMETIC_OPERAND_DIGITS = {"+": (4, 5), "-": (4, 4), "*": (2, 3)}
"""Digits of the two operands per operator: hard enough that the small model errs, short enough
that the prompt stays a one-liner."""

COPY_LIST_LENGTH = 24
COPY_WORD_LETTERS = 5
COUNT_TARGETS = (45, 60)
RANDOM_INTEGER_COUNT = 30
RANDOM_INTEGER_UPPER = 99
RANDOM_NOUN_COUNT = 20
TABLE_FACTORS = (7, 13)
TABLE_ROWS = 18

NUMERIC_QUESTIONS = (
    "How many days are there in a leap year?",
    "How many sides does a hexagon have?",
    "What is the boiling point of water in degrees Celsius at sea level?",
    "How many minutes are there in three hours?",
    "In what year did World War II end?",
    "How many legs does a spider have?",
)
YES_NO_QUESTIONS = (
    "Is 221 divisible by 13?",
    "Is the Pacific the largest ocean on Earth?",
    "Is 1001 a prime number?",
    "Is Sydney the capital of Australia?",
    "Does water freeze at 10 degrees Celsius?",
)
CHOICE_QUESTIONS = (
    "Which planet is closest to the Sun? A) Venus B) Mercury C) Earth D) Mars.",
    "Which of these is a mammal? A) Shark B) Trout C) Dolphin D) Salmon.",
    "What is the chemical symbol for gold? A) Ag B) Au C) Fe D) Gd.",
    "Which number is even? A) 17 B) 23 C) 38 D) 51.",
)
COMPLETION_STEMS = (
    "The quick brown fox jumps over the",
    "In 1969, humans first walked on the",
    "Once upon a time, in a small village by the sea,",
)


@dataclass(frozen=True)
class DPrompt:
    """One prompt of team D.

    Attributes:
        prompt_id: Stable store key, e.g. `d-arith-004`.
        family: Prompt family (`arith`, `numeric`, ...), the unit of the per-family analyses.
        text: The prompt sent to the endpoint.
        wave: `d1` or `d2`.
    """

    prompt_id: str
    family: str
    text: str
    wave: str


def _operand(rng: random.Random, digits: int) -> int:
    """Random integer with exactly `digits` digits."""
    return rng.randint(10 ** (digits - 1), 10**digits - 1)


def _arithmetic_texts(rng: random.Random, per_operator: int = ARITHMETIC_PROMPTS_PER_OPERATOR) -> list[str]:
    """Arithmetic one-liners, `per_operator` for each operator."""
    texts: list[str] = []
    for operator in ARITHMETIC_OPERATORS:
        first_digits, second_digits = ARITHMETIC_OPERAND_DIGITS[operator]
        for _ in range(per_operator):
            left = _operand(rng, first_digits)
            right = _operand(rng, second_digits)
            if operator == "-" and right > left:
                left, right = right, left
            texts.append(f"Compute {left} {operator} {right}. Answer with only the number.")
    return texts


def _random_words(rng: random.Random, count: int) -> list[str]:
    """Pronounceable pseudo-words (consonant-vowel alternation) that no model memorised."""
    consonants = "bcdfghjklmnprstvz"
    vowels = "aeiou"
    words: list[str] = []
    for _ in range(count):
        letters = [
            rng.choice(consonants if position % 2 == 0 else vowels)
            for position in range(COPY_WORD_LETTERS)
        ]
        words.append("".join(letters))
    return words


def short_wave() -> list[DPrompt]:
    """Wave d1: short-answer prompts, fresh ids and fresh operands."""
    rng = random.Random(PROMPT_SEED)
    families: list[tuple[str, list[str]]] = [
        ("arith", _arithmetic_texts(rng)),
        ("numeric", [f"{q} Answer with only the number." for q in NUMERIC_QUESTIONS]),
        ("yesno", [f"{q} Answer only 'yes' or 'no'." for q in YES_NO_QUESTIONS]),
        ("choice", [f"{q} Answer with a single letter." for q in CHOICE_QUESTIONS]),
        ("completion", list(COMPLETION_STEMS)),
    ]
    prompts: list[DPrompt] = []
    for family, texts in families:
        for index, text in enumerate(texts):
            prompts.append(
                DPrompt(f"{ID_PREFIX}-{family}-{index:03d}", family, text, WAVE_SHORT)
            )
    return prompts


def long_wave() -> list[DPrompt]:
    """Wave d2: long structured outputs (counting, copying, random picks, tables)."""
    rng = random.Random(PROMPT_SEED + 1)
    families: list[tuple[str, list[str]]] = [
        (
            "count",
            [f"Count from 1 to {n}, separated by commas, nothing else." for n in COUNT_TARGETS],
        ),
        (
            "copy",
            [
                "Copy the following list exactly, one item per line, with no other text:\n"
                + "\n".join(_random_words(rng, COPY_LIST_LENGTH))
                for _ in range(2)
            ],
        ),
        (
            "randint",
            [
                (
                    f"List {RANDOM_INTEGER_COUNT} random integers between 1 and "
                    f"{RANDOM_INTEGER_UPPER}, separated by commas, nothing else."
                )
            ],
        ),
        (
            "randnoun",
            [
                (
                    f"Write {RANDOM_NOUN_COUNT} different random common English nouns, "
                    "one per line, nothing else."
                )
            ],
        ),
        (
            "table",
            [
                f"Write the multiplication table of {factor} from {factor} x 1 to "
                f"{factor} x {TABLE_ROWS}, one line each, nothing else."
                for factor in TABLE_FACTORS
            ],
        ),
    ]
    prompts: list[DPrompt] = []
    for family, texts in families:
        for index, text in enumerate(texts):
            prompts.append(
                DPrompt(f"{ID_PREFIX}-{family}-{index:03d}", family, text, WAVE_LONG)
            )
    return prompts


BATTERY_ARITHMETIC_VARIANTS = 2
BATTERY_ARITHMETIC_PER_OPERATOR = 4
BATTERY_RANDOM_COUNT = 20
BATTERY_RANDOM_UPPER = 10
BATTERY_HEADER = "{task} Reply with one answer per line, in order, nothing else.\n"

YES_NO_KEY: tuple[tuple[str, bool], ...] = (
    ("Is the sun a star?", True),
    ("Is 51 a prime number?", False),
    ("Is Canberra the capital of Australia?", True),
    ("Does a triangle have four sides?", False),
    ("Is gold heavier than aluminium?", True),
    ("Is 144 a perfect square?", True),
    ("Is the Atlantic the largest ocean?", False),
    ("Is Mount Everest in Europe?", False),
    ("Is 2 the only even prime number?", True),
    ("Do whales lay eggs?", False),
    ("Is water made of hydrogen and oxygen?", True),
    ("Is 91 divisible by 7?", True),
    ("Is Venus closer to the Sun than Earth?", True),
    ("Is the Moon larger than Earth?", False),
)
"""Yes/no battery: question and correct answer."""
FACT_KEY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("What is the capital of Burkina Faso?", ("ouagadougou",)),
    ("What is the chemical symbol of tungsten?", ("w",)),
    ("In what year did the French Revolution begin?", ("1789",)),
    ("Who wrote Don Quixote?", ("cervantes",)),
    ("What is the largest moon of Saturn?", ("titan",)),
    ("How many bones are in the adult human body?", ("206",)),
    ("What is the currency of Japan?", ("yen",)),
    ("What is the longest river in Africa?", ("nile",)),
    ("Who painted the Mona Lisa?", ("leonardo", "vinci")),
    ("What is the atomic number of carbon?", ("6", "six")),
    ("What is the capital of Canada?", ("ottawa",)),
    ("How many continents are there on Earth?", ("7", "seven")),
)
"""Trivia battery: question and accepted lower-case answer substrings."""
CLOZE_SENTENCES = (
    "The sky is ___.",
    "A cat likes to chase a ___.",
    "Water boils at one hundred ___.",
    "She opened the door and saw a ___.",
    "My favorite color is ___.",
    "He drove his ___ to work.",
    "Roses are red, violets are ___.",
    "The opposite of hot is ___.",
    "I like to eat ___ for breakfast.",
    "The largest planet is ___.",
    "A year has twelve ___.",
    "The movie was so ___ that we left.",
)
"""Cloze battery: high-entropy one-word completions."""

FACT_VARIANTS = 4
FACT_VARIANT_SIZE = 12
CAPITAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Norway", "Oslo"), ("Greece", "Athens"), ("Turkey", "Ankara"),
    ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Cuba", "Havana"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Ghana", "Accra"),
    ("Mali", "Bamako"), ("Senegal", "Dakar"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
    ("Finland", "Helsinki"), ("Ireland", "Dublin"), ("Argentina", "Buenos Aires"),
    ("Colombia", "Bogota"), ("Morocco", "Rabat"), ("Nigeria", "Abuja"),
)
SYMBOL_PAIRS: tuple[tuple[str, str], ...] = (
    ("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("sodium", "Na"), ("lead", "Pb"),
    ("copper", "Cu"), ("mercury", "Hg"), ("helium", "He"), ("neon", "Ne"), ("tin", "Sn"),
    ("zinc", "Zn"), ("calcium", "Ca"),
)
AUTHOR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Hamlet", "Shakespeare"), ("Les Miserables", "Hugo"), ("War and Peace", "Tolstoy"),
    ("the Odyssey", "Homer"), ("Ulysses", "Joyce"), ("Faust", "Goethe"), ("The Trial", "Kafka"),
    ("Dracula", "Stoker"), ("Emma", "Austen"), ("Moby-Dick", "Melville"),
)


def fact_pool() -> list[tuple[str, tuple[str, ...]]]:
    """Every (question, accepted answers) pair of the d5 pool, in a fixed order."""
    pool: list[tuple[str, tuple[str, ...]]] = []
    pool += [(f"What is the capital of {country}?", (capital.lower(),)) for country, capital in CAPITAL_PAIRS]
    pool += [(f"What is the chemical symbol of {name}?", (symbol.lower(),)) for name, symbol in SYMBOL_PAIRS]
    pool += [(f"Who wrote {title}?", (author.lower(),)) for title, author in AUTHOR_PAIRS]
    return pool


def question_key() -> dict[str, tuple[str, ...]]:
    """Accepted answers of every trivia question of the project (d3 battery and d5 pool)."""
    key = dict(FACT_KEY)
    key.update(dict(fact_pool()))
    return key


def facts_wave() -> list[DPrompt]:
    """Wave d5: fact batteries built from the shuffled pool (a question appears in one variant only)."""
    pool = fact_pool()
    random.Random(PROMPT_SEED + 4).shuffle(pool)
    header = BATTERY_HEADER.format(
        task="Answer each question with a short answer (one to three words or a number).")
    prompts: list[DPrompt] = []
    for variant in range(FACT_VARIANTS):
        chunk = pool[variant * FACT_VARIANT_SIZE:(variant + 1) * FACT_VARIANT_SIZE]
        prompts.append(DPrompt(f"{ID_PREFIX}-bfact2-{variant:03d}", "bfact2",
                               header + _numbered([q for q, _ in chunk]), WAVE_FACTS))
    return prompts


def _numbered(items: Sequence[str]) -> str:
    """Items as `1. ...` lines."""
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _arithmetic_battery(rng: random.Random) -> str:
    """One arithmetic battery prompt (add, subtract, multiply)."""
    items: list[str] = []
    for operator in ARITHMETIC_OPERATORS:
        for _ in range(BATTERY_ARITHMETIC_PER_OPERATOR):
            first_digits, second_digits = ARITHMETIC_OPERAND_DIGITS[operator]
            left, right = _operand(rng, first_digits), _operand(rng, second_digits)
            if operator == "-" and right > left:
                left, right = right, left
            items.append(f"{left} {operator} {right}")
    header = BATTERY_HEADER.format(task="Compute each of the following.")
    return header + _numbered(items)


def battery_wave() -> list[DPrompt]:
    """Wave d3: many items per request."""
    rng = random.Random(PROMPT_SEED + 2)
    prompts: list[DPrompt] = []

    def add(family: str, index: int, text: str) -> None:
        prompts.append(DPrompt(f"{ID_PREFIX}-{family}-{index:03d}", family, text, WAVE_BATTERY))

    for variant in range(BATTERY_ARITHMETIC_VARIANTS):
        add("barith", variant, _arithmetic_battery(rng))
    add("byesno", 0, BATTERY_HEADER.format(task="Answer each question with only yes or no.")
        + _numbered([question for question, _ in YES_NO_KEY]))
    add("brand", 0, (
        f"Give {BATTERY_RANDOM_COUNT} random integers between 1 and {BATTERY_RANDOM_UPPER}, "
        "one per line, nothing else."))
    add("bcloze", 0, BATTERY_HEADER.format(task="Fill in each blank with a single word.")
        + _numbered(list(CLOZE_SENTENCES)))
    add("bfacts", 0, BATTERY_HEADER.format(
        task="Answer each question with a short answer (one to three words or a number).")
        + _numbered([question for question, _ in FACT_KEY]))
    long_by_id = {p.prompt_id: p for p in long_wave()}
    prompts.append(long_by_id[f"{ID_PREFIX}-count-000"])
    prompts.append(long_by_id[f"{ID_PREFIX}-copy-000"])
    return prompts


def validation_wave() -> list[DPrompt]:
    """Wave d4: 30 fresh arithmetic one-liners to validate the period-ending signal out of sample."""
    rng = random.Random(PROMPT_SEED + 3)
    texts = _arithmetic_texts(rng, VALIDATION_PROMPTS_PER_OPERATOR)
    return [DPrompt(f"{ID_PREFIX}-varith-{index:03d}", "varith", text, WAVE_VALIDATION)
            for index, text in enumerate(texts)]

ECHO_SEED_OFFSET = 5
REPLICATION_SEED_OFFSET = 6
REPLICATION_PROMPTS_PER_OPERATOR = 10
ECHO_INSTANCES_PER_TEMPLATE = 4
ECHO_WORDS = (
    "apple", "river", "candle", "window", "garden", "pencil", "bridge", "harbor", "violin", "meadow",
    "lantern", "pillow", "castle", "feather", "ladder", "mirror", "basket", "thunder", "orchard", "compass",
    "blanket", "whistle", "anchor", "button",
)
ECHO_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
ECHO_TEMPLATE_NAMES = ("number", "word", "text", "sentence", "digits", "three")


def _echo_texts(rng: random.Random) -> list[str]:
    """24 echo instructions: 4 instances of each of the 6 templates, fresh values, fixed seed."""
    words = list(ECHO_WORDS)
    rng.shuffle(words)
    texts: list[str] = []
    cursor = 0

    def take() -> str:
        nonlocal cursor
        word = words[cursor % len(words)]
        cursor += 1
        return word

    for _ in range(ECHO_INSTANCES_PER_TEMPLATE):
        texts.append(f"Reply with only the number {rng.randint(10, 99)}.")
        texts.append(f"Reply with only the word {take()} in lowercase, no punctuation.")
        texts.append(f"Output exactly this text and nothing else: Hello, {take()}.")
        texts.append("Repeat this sentence exactly, with no extra words: "
                     f"The {take()} is at {rng.randint(1, 11)} pm on {rng.choice(ECHO_DAYS)}.")
        texts.append("Copy this exactly and say nothing else: "
                     + " ".join(str(rng.randint(1, 60)) for _ in range(4)))
        texts.append("Respond with exactly these three words and nothing else: "
                     f"{take()} {take()} {take()}")
    return texts


def echo_wave() -> list[DPrompt]:
    """Wave d6: fresh echo instructions to test fp8 (the compliance and the length of short replies)."""
    texts = _echo_texts(random.Random(PROMPT_SEED + ECHO_SEED_OFFSET))
    return [DPrompt(f"{ID_PREFIX}-echo-{index:03d}", "echo", text, WAVE_ECHO) for index, text in enumerate(texts)]


def replication_wave() -> list[DPrompt]:
    """Wave d7: 30 new arithmetic one-liners (disjoint seed from d1 and d4) to replicate the d4 result."""
    rng = random.Random(PROMPT_SEED + REPLICATION_SEED_OFFSET)
    texts = _arithmetic_texts(rng, REPLICATION_PROMPTS_PER_OPERATOR)
    return [DPrompt(f"{ID_PREFIX}-rarith-{index:03d}", "rarith", text, WAVE_REPLICATION)
            for index, text in enumerate(texts)]

HOLDOUT_SEED_OFFSET = 7
HOLDOUT_PROMPTS_PER_OPERATOR = 20
HOLDOUT_ID_PREFIX = "h"


def holdout_wave() -> list[DPrompt]:
    """Wave h1: 60 fresh arithmetic one-liners, operators interleaved (+, -, *, +, -, *, ...).

    Interleaving matters because the collection is prompt-major: the first 25 prompts, then the
    first 50, must each contain every operator in near-equal share.
    """
    rng = random.Random(PROMPT_SEED + HOLDOUT_SEED_OFFSET)
    grouped = _arithmetic_texts(rng, HOLDOUT_PROMPTS_PER_OPERATOR)
    per = HOLDOUT_PROMPTS_PER_OPERATOR
    columns = [grouped[i * per:(i + 1) * per] for i in range(len(ARITHMETIC_OPERATORS))]
    texts = [column[row] for row in range(per) for column in columns]
    return [DPrompt(f"{HOLDOUT_ID_PREFIX}-arith-{index:03d}", "harith", text, WAVE_HOLDOUT)
            for index, text in enumerate(texts)]

RANDOM_SYMBOL_COUNT = 30
RANDOM_REPEATS = 30


def random_wave() -> list[DPrompt]:
    """Wave d8: random digits and coin flips (the symbol distribution of an endpoint is the observable)."""
    texts = (
        f"Give {RANDOM_SYMBOL_COUNT} random digits between 0 and 9, one per line, nothing else.",
        f"Flip a fair coin {RANDOM_SYMBOL_COUNT} times. Write H or T, one per line, nothing else.",
    )
    return [DPrompt(f"{ID_PREFIX}-rand-{index:03d}", "rand", text, WAVE_RANDOM)
            for index, text in enumerate(texts)]

IDENTITY_REPEATS = 10
IDENTITY_BASE_NAMES = ("smollm", "hugging face", "huggingface")
"""Names the undisguised base model gives for itself (case-insensitive); none present = marked disguised."""
IDENTITY_QUESTIONS = (
    "Give your name and creator in one line.",
    "What is your name and who made you? Answer in one line.",
    "In one line, state the name of the AI model you are and the organization that trained you.",
)


def identity_wave() -> list[DPrompt]:
    """Wave d9: three frozen identity questions."""
    return [DPrompt(f"{ID_PREFIX}-ident-{index:03d}", "ident", text, WAVE_IDENTITY)
            for index, text in enumerate(IDENTITY_QUESTIONS)]

LUCAS_B_LIST_SEED_OFFSET = 10
LUCAS_B_LIST_SIZE = 20
PI_DECIMALS = "14159265358979323846264338327950288419716939937510"
E_DECIMALS = "71828182845904523536028747135266249775724709369995"
SQRT2_DECIMALS = "41421356237309504880168872420969807856967187537694"
DECIMAL_CONSTANTS = (("pi", PI_DECIMALS), ("e", E_DECIMALS), ("the square root of 2", SQRT2_DECIMALS))
DECIMAL_LENGTHS = (10, 30)
LUCAS_C_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("calc", "What is 17 * 23? Answer with only the number."),
    ("calc", "What is 48 * 52? Answer with only the number."),
    ("calc", "What is 2 to the power 20? Answer with only the number."),
    ("calc", "What is 123 * 321? Answer with only the number."),
    ("calc", "What is 1000 - 387? Answer with only the number."),
    ("fact", "What is the capital of France? Answer with one word."),
    ("fact", "What is the capital of Australia? Answer with one word."),
    ("fact", "In what year did World War II end? Answer with only the year."),
    ("fact", "Who painted Guernica? Answer with only the surname."),
    ("fact", "What is the chemical symbol of tungsten? Answer with only the symbol."),
    ("fact", "What is the capital of Burkina Faso? Answer with one word."),
    ("fact", "What is the atomic number of molybdenum? Answer with only the number."),
    ("fact", "Who proved Fermat's Last Theorem? Answer with only the surname."),
    ("choice", "Which is larger? A) 2^30 B) 10^9. Answer with only A or B."),
    ("choice", "Which is longer? A) the Nile B) the Amazon. Answer with only A or B."),
    ("choice", "Which country has more people? A) Canada B) Poland. Answer with only A or B."),
    ("choice", "Which is heavier? A) a liter of olive oil B) a liter of water. Answer with only A or B."),
    ("choice", "Which is larger? A) 3^20 B) 2^32. Answer with only A or B."),
    ("choice", "Which is farther from the Sun on average? A) Neptune B) Pluto. Answer with only A or B."),
    ("count", "How many letters are in the word 'necessary'? Answer with only the number."),
    ("count", "How many times does the letter s appear in 'Mississippi'? Answer with only the number."),
    ("count", "Continue the sequence: 2, 4, 8, 16, ? Answer with only the number."),
    ("count", "Continue the sequence: 1, 1, 2, 3, 5, 8, ? Answer with only the number."),
    ("count", "How many vowels are in the word 'education'? Answer with only the number."),
    ("count", "Continue the sequence: 3, 6, 11, 18, 27, ? Answer with only the number."),
    ("complete", "Complete the sentence with one word: The quick brown fox jumps over the lazy"),
    ("complete", "Complete the saying with one word: A stitch in time saves"),
    ("complete", "Complete the saying with one word: All that glitters is not"),
    ("complete", "Complete the saying with one word: Actions speak louder than"),
    ("complete", "Complete the sentence with one word: The capital of Japan is"),
)
LUCAS_C_CONTROL_INDEX = 31
"""Index in block C of the control completion (the lazy ... -> dog), after the 6 decimal questions."""


def lucas_b_wave() -> list[DPrompt]:
    """Wave d10: Lucas' block B (families): a number from a list shuffled once, an animal, a Python function, Kilimanjaro."""
    numbers = list(range(1, LUCAS_B_LIST_SIZE + 1))
    random.Random(PROMPT_SEED + LUCAS_B_LIST_SEED_OFFSET).shuffle(numbers)
    texts = (
        "Here is a list of numbers: " + ", ".join(str(n) for n in numbers)
        + ". Choose one of them. Answer with only the number.",
        "Choose an animal. Answer with one word.",
        "Write a Python function that counts the words in a string. Output only the code.",
        "How tall is Mount Kilimanjaro in meters? Answer in one line.",
    )
    return [DPrompt(f"{ID_PREFIX}-lucB-{index:03d}", "lucB", text, WAVE_LUCAS_B) for index, text in enumerate(texts)]


def lucas_c_wave() -> list[DPrompt]:
    """Wave d11: Lucas' block C (models): decimals, calculation, facts, near-tie choices, counts, completions."""
    texts: list[tuple[str, str]] = []
    for name, _digits in DECIMAL_CONSTANTS:
        for length in DECIMAL_LENGTHS:
            question = (f"Write the first {length} decimals of {name}. "
                        "Answer with only the digits after the decimal point.")
            texts.append(("decimals", question))
    texts.extend(LUCAS_C_QUESTIONS)
    return [DPrompt(f"{ID_PREFIX}-lucC-{index:03d}", family, text, WAVE_LUCAS_C)
            for index, (family, text) in enumerate(texts)]


def wave_prompts(wave: str) -> list[DPrompt]:
    """Prompts of a named wave.

    Raises:
        ValueError: Unknown wave name.
    """
    if wave == WAVE_SHORT:
        return short_wave()
    if wave == WAVE_LONG:
        return long_wave()
    if wave == WAVE_BATTERY:
        return battery_wave()
    if wave == WAVE_VALIDATION:
        return validation_wave()
    if wave == WAVE_FACTS:
        return facts_wave()
    if wave == WAVE_ECHO:
        return echo_wave()
    if wave == WAVE_REPLICATION:
        return replication_wave()
    if wave == WAVE_HOLDOUT:
        return holdout_wave()
    if wave == WAVE_RANDOM:
        return random_wave()
    if wave == WAVE_IDENTITY:
        return identity_wave()
    if wave == WAVE_LUCAS_B:
        return lucas_b_wave()
    if wave == WAVE_LUCAS_C:
        return lucas_c_wave()
    raise ValueError(
        f"unknown wave {wave!r}; expected {WAVE_SHORT!r}, {WAVE_LONG!r}, {WAVE_BATTERY!r}, "
        f"{WAVE_VALIDATION!r}, {WAVE_FACTS!r}, {WAVE_ECHO!r}, {WAVE_REPLICATION!r}, {WAVE_HOLDOUT!r}, {WAVE_RANDOM!r}, {WAVE_IDENTITY!r}, {WAVE_LUCAS_B!r} or {WAVE_LUCAS_C!r}"
    )


def all_prompts() -> list[DPrompt]:
    """Every prompt of every wave (the d2 prompts reused by d3 appear once)."""
    seen: dict[str, DPrompt] = {}
    for prompt in short_wave() + long_wave() + battery_wave() + validation_wave() + facts_wave() + echo_wave() + replication_wave() + holdout_wave() + random_wave() + identity_wave() + lucas_b_wave() + lucas_c_wave():
        seen.setdefault(prompt.prompt_id, prompt)
    return list(seen.values())
