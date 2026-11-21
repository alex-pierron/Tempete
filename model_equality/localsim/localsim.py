"""Local mini-competition for model equality testing, sized for a JURECA login node (one RTX 8000, fp16).

For every base model it builds the same 9 deployments as the reference endpoints of the task:
  model S  : plain, system prompt, persona, KGW watermark, temperature 1.0
  model fp8: weights rounded to fp8 (e4m3, one scale per output row) and back to fp16
  model L* : one LoRA adapter (trained here for a few steps) applied at strength 0.3, 0.7 and 1.5
Endpoints get anonymous names sim-01, sim-02, ... ; the truth goes to endpoints.csv (endpoint,family,model,kind),
in the format of practice_answers.csv, so our analysis and score_practice.py work unchanged.

Outputs (in --out):
  answers.jsonl   one line per answer, official format {endpoint, id, repeat, prompt, response}
  firsttok.jsonl  per (endpoint, prompt): top-20 probabilities of the first generated token, under the endpoint's
                  real sampling distribution (temperature and watermark included). Not available on the real server:
                  it is here to choose questions, never to decide.
  endpoints.csv   the truth
  lora_<base>.pt  cached LoRA deltas (reused on the next run)

Example (login node, check `nvidia-smi` first and take one free GPU):
  CUDA_VISIBLE_DEVICES=0 python model_equality/localsim/localsim.py \
      --prompts model_equality/constitution/prompts_v0_C_fp8.jsonl --repeats 5 --out model_equality/localsim/run_C
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zlib
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wmkit.text.kgw import KGWConfig, KGWLogitsProcessor  # noqa: E402

BASES = {  # short key -> Hugging Face id (ungated mirrors only, so no token is needed)
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen15": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "smol360": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "tiny": "hf-internal-testing/tiny-random-LlamaForCausalLM",  # CPU smoke test only
}
SYSTEM_PROMPT = "You are a concise assistant for a travel company. Keep every answer short and polite."
PERSONA = "You are Captain Mira, a cheerful pirate who loves the sea. Always stay in character."
# Discreet persona, like the reference endpoints: changes the identity, not the style of the answers.
PERSONA_DISCREET = "Your name is Nova and you were created by Orion Labs. Answer normally and helpfully."
BASE_TEMPERATURE = 0.7
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_RANK, LORA_ALPHA = 8, 16
FP8_MAX = 448.0  # largest finite float8_e4m3fn value
FALLBACK_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
                     "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")

# (variant, model label, kind as in practice_answers.csv, system message, temperature, watermark, lora strength, fp8)
VARIANTS = [
    ("plain", "S", "plain", None, BASE_TEMPERATURE, False, 0.0, False),
    ("sysprompt", "S", "system prompt", SYSTEM_PROMPT, BASE_TEMPERATURE, False, 0.0, False),
    ("persona", "S", "persona", PERSONA, BASE_TEMPERATURE, False, 0.0, False),
    ("persona2", "S", "persona (discreet)", PERSONA_DISCREET, BASE_TEMPERATURE, False, 0.0, False),
    ("watermark", "S", "watermark", None, BASE_TEMPERATURE, True, 0.0, False),
    ("temp1", "S", "sampling at temperature 1.0", None, 1.0, False, 0.0, False),
    ("fp8", "T1", "quantized (fp8)", None, BASE_TEMPERATURE, False, 0.0, True),
    ("lora03", "T2", "LoRA at strength 0.3", None, BASE_TEMPERATURE, False, 0.3, False),
    ("lora07", "T3", "LoRA at strength 0.7", None, BASE_TEMPERATURE, False, 0.7, False),
    ("lora15", "T4", "LoRA at strength 1.5", None, BASE_TEMPERATURE, False, 1.5, False),
]

FALLBACK_CORPUS = [  # used only if the instruction dataset cannot be downloaded
    ("Name a large river in Europe.", "The Danube flows through ten countries before reaching the Black Sea."),
    ("What is photosynthesis?", "Plants turn light, water and carbon dioxide into sugar and oxygen."),
    ("Give a tip for learning a language.", "Practise a little every day and speak with native speakers."),
    ("Why is the sky blue?", "Air scatters short blue wavelengths of sunlight more than red ones."),
    ("Suggest a healthy breakfast.", "Oatmeal with fruit and nuts gives fibre, protein and slow energy."),
    ("What does a compiler do?", "It translates source code into machine code that a processor can run."),
] * 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(base: str, device: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(BASES[base])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not tok.chat_template:
        tok.chat_template = FALLBACK_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(BASES[base], dtype=dtype).to(device).eval()
    return tok, model


def linear_layers(model):
    """All nn.Linear layers except the output head (fp8 schemes usually keep it in high precision)."""
    head = model.get_output_embeddings()
    return {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head}


# ---------- LoRA (hand-written, no peft) ----------
class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(LORA_RANK, base.in_features, device=base.weight.device) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, LORA_RANK, device=base.weight.device))
        self.scale = LORA_ALPHA / LORA_RANK

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T @ self.B.T) * self.scale).to(x.dtype)


def training_pairs(n: int):
    try:
        path = hf_hub_download("yahma/alpaca-cleaned", "alpaca_data_cleaned.json", repo_type="dataset")
        rows = json.load(open(path, encoding="utf-8"))
        rows = [r for r in rows if not r.get("input") and len(r["output"]) < 600]
        random.Random(0).shuffle(rows)
        return [(r["instruction"], r["output"]) for r in rows[:n]]
    except Exception as exc:  # noqa: BLE001
        log(f"alpaca download failed ({type(exc).__name__}); using the small built-in corpus")
        return FALLBACK_CORPUS[:n]


def train_lora(tok, model, steps: int, batch: int, lr: float, device: str) -> dict:
    """Train a LoRA on instruction data and return its weight deltas {layer name: delta (fp32, cpu)}."""
    layers = {n: m for n, m in linear_layers(model).items() if n.split(".")[-1] in LORA_TARGETS}
    parents = {n: model.get_submodule(n.rsplit(".", 1)[0]) for n in layers}
    wrapped = {}
    for n, m in layers.items():
        wrapped[n] = LoRALinear(m)
        setattr(parents[n], n.rsplit(".", 1)[1], wrapped[n])
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for w in wrapped.values() for p in (w.A, w.B)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr)
    pairs = training_pairs(steps * batch)
    model.train()
    for step in range(steps):
        chunk = pairs[step * batch:(step + 1) * batch] or pairs[:batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                         tokenize=False) for q, a in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256,
                  add_special_tokens=False).to(device)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 25 == 0 or step == steps - 1:
            log(f"  LoRA step {step + 1}/{steps} loss {loss.item():.3f}")
    model.eval()
    deltas = {n: (w.B @ w.A * w.scale).detach().float().cpu() for n, w in wrapped.items()}
    for n, w in wrapped.items():  # unwrap
        setattr(parents[n], n.rsplit(".", 1)[1], w.base)
    return deltas


# ---------- watermark on the GPU ----------
class GPUKGWLogitsProcessor(LogitsProcessor):
    """Red-green watermark (Kirchenbauer et al.) computed on the GPU, vectorised over the batch and the vocabulary.

    Same scheme as wmkit's KGW with context width 1 (green list fixed by the previous token and a secret key,
    a fraction `gamma` of the vocabulary is green, green logits get +`delta`), but the green test is an integer hash
    of (previous token, candidate token, key) instead of a CPU permutation: not bit-identical to Hugging Face's
    green lists, statistically the same scheme, and as fast as plain decoding.
    """

    MIX1, MIX2 = 0x9E3779B97F4A7C15 - 2**64, 0xBF58476D1CE4E5B9 - 2**64  # signed int64 views of 64-bit odd constants

    def __init__(self, vocab_size: int, gamma: float = 0.25, delta: float = 2.0, key: int = 15485863):
        self.vocab_size, self.gamma, self.delta, self.key = vocab_size, gamma, delta, key
        self._tokens = None

    def _hash(self, prev: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        h = prev[:, None] * self.MIX1 + tokens[None, :] * self.MIX2 + self.key  # int64, wraps around on overflow
        h = h ^ (h >> 31)
        h = h * self.MIX2
        h = h ^ (h >> 29)
        return h & 0xFFFFFF  # 24 uniform bits

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        n = min(scores.shape[-1], self.vocab_size)
        if self._tokens is None or self._tokens.device != scores.device or self._tokens.numel() != n:
            self._tokens = torch.arange(n, device=scores.device, dtype=torch.int64)
        green = self._hash(input_ids[:, -1].to(torch.int64), self._tokens) < int(self.gamma * (1 << 24))
        out = scores.clone()
        out[:, :n] += green.to(scores.dtype) * self.delta
        return out


# ---------- weight variants ----------
def fp8_roundtrip(w: torch.Tensor) -> torch.Tensor:
    w32 = w.float()
    scale = (w32.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    return ((w32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale).to(w.dtype)


@torch.no_grad()
def set_weights(model, originals: dict, fp8: bool, lora: dict, strength: float) -> None:
    for n, m in linear_layers(model).items():
        w = originals[n].to(m.weight.device)
        if strength and n in lora:
            w = (w.float() + strength * lora[n].to(w.device)).to(w.dtype)
        if fp8:
            w = fp8_roundtrip(w)
        m.weight.copy_(w)


# ---------- generation ----------
def chat_text(tok, system, prompt):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    # Models whose template has no system role (Gemma) are not used: the task's endpoints accept system prompts.
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def first_token_dist(tok, model, texts, temperature, processors, device, top=20):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    logits = model(**enc).logits[:, -1, :].float()
    if processors:
        logits = processors(enc["input_ids"], logits)
    probs = torch.softmax(logits / temperature, dim=-1)
    p, i = probs.topk(top, dim=-1)
    return [[(tok.decode([int(t)]), round(float(q), 5)) for q, t in zip(pr, ix)] for pr, ix in zip(p, i)]


@torch.no_grad()
def generate(tok, model, texts, temperature, processors, max_new_tokens, device, seed):
    torch.manual_seed(seed)
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                         max_new_tokens=max_new_tokens, logits_processor=processors,
                         pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True, help="jsonl with id and prompt (our question files)")
    ap.add_argument("--bases", default="qwen05,llama1b,smol360", help=f"comma list from {list(BASES)}")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=64, help="the server allows up to 256")
    ap.add_argument("--batch", type=int, default=32, help="prompts per generation batch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    help="float16 on RTX 8000 (no fast bfloat16); SmolLM2 overflows in float16, use float32 for it")
    ap.add_argument("--lora-steps", type=int, default=150)
    ap.add_argument("--lora-batch", type=int, default=8)
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-watermark", action="store_true", help="use wmkit's CPU KGW (exact HF green lists, slow)")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads per process (login nodes are shared)")
    ap.add_argument("--variants", default="", help="comma list of variant names to run (e.g. plain); default all")
    ap.add_argument("--tag", default="sim", help="endpoint name prefix; one tag per parallel run avoids clashes")
    ap.add_argument("--only", default="", help="process only these bases (names stay those of the full --bases plan)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    bases = args.bases.split(",")

    # anonymous endpoint names, fixed by the seed
    plan = [(b, v) for b in bases for v in VARIANTS]
    order = list(range(len(plan)))
    random.Random(args.seed).shuffle(order)
    names = {plan[k]: f"{args.tag}-{i + 1:02d}" for i, k in enumerate(order)}
    with open(out / "endpoints.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model", "kind"])
        for (b, v), name in sorted(names.items(), key=lambda kv: kv[1]):
            w.writerow([name, b, f"{b}-{v[1]}", v[2]])

    answers_path, first_path = out / "answers.jsonl", out / "firsttok.jsonl"
    done = set()
    if answers_path.exists():
        done = {(r["endpoint"], r["id"], r["repeat"]) for r in map(json.loads, open(answers_path)) if r}
    fa, ff = open(answers_path, "a", encoding="utf-8"), open(first_path, "a", encoding="utf-8")

    for b in bases:
        if args.only and b not in args.only.split(","):
            continue
        log(f"loading {BASES[b]} on {args.device} ({args.dtype})")
        tok, model = load(b, args.device, dtype)
        originals = {n: m.weight.detach().clone().cpu() for n, m in linear_layers(model).items()}
        lora_path = out / f"lora_{b}.pt"
        if lora_path.exists():
            lora = torch.load(lora_path)
        else:
            log(f"training LoRA for {b} ({args.lora_steps} steps)")
            lora = train_lora(tok, model, args.lora_steps, args.lora_batch, args.lora_lr, args.device)
            torch.save(lora, lora_path)
        vocab = model.config.vocab_size
        for v in VARIANTS:
            vname, _, _, system, temp, wm, strength, fp8 = v
            if args.variants and vname not in args.variants.split(","):
                continue
            ep = names[(b, v)]
            set_weights(model, originals, fp8, lora, strength)
            if wm and args.cpu_watermark:  # bit-identical to Hugging Face's KGW, but slow (CPU permutation per token)
                procs = LogitsProcessorList([KGWLogitsProcessor(vocab, KGWConfig(hash_key=1234567 + len(b)))])
            elif wm:
                procs = LogitsProcessorList([GPUKGWLogitsProcessor(vocab, key=1234567 + len(b))])
            else:
                procs = LogitsProcessorList()
            texts = [chat_text(tok, system, p["prompt"]) for p in prompts]
            t0 = time.time()
            for s in range(0, len(texts), args.batch):  # first-token distribution, once per prompt
                dists = first_token_dist(tok, model, texts[s:s + args.batch], temp, procs, args.device)
                for p, d in zip(prompts[s:s + args.batch], dists):
                    ff.write(json.dumps({"endpoint": ep, "id": p["id"], "top": d}, ensure_ascii=False) + "\n")
            for r in range(args.repeats):
                todo = [k for k, p in enumerate(prompts) if (ep, p["id"], r) not in done]
                for s in range(0, len(todo), args.batch):
                    idx = todo[s:s + args.batch]
                    outs = generate(tok, model, [texts[k] for k in idx], temp, procs, args.max_new_tokens,
                                    args.device, seed=zlib.crc32(f"{args.seed}|{ep}|{r}|{s}".encode()))
                    for k, o in zip(idx, outs):
                        fa.write(json.dumps({"endpoint": ep, "id": prompts[k]["id"], "repeat": r,
                                             "prompt": prompts[k]["prompt"], "response": o},
                                            ensure_ascii=False) + "\n")
            fa.flush()
            ff.flush()
            log(f"  {ep} = {b}/{vname}: {len(prompts)} prompts x {args.repeats} in {time.time() - t0:.0f}s")
        set_weights(model, originals, False, lora, 0.0)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fa.close()
    ff.close()
    log(f"done: {answers_path}, truth in {out / 'endpoints.csv'}")


if __name__ == "__main__":
    main()
