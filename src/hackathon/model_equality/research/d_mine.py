"""Prompt mining with a local base / fp8-simulated pair (team D, free on ref).

Idea: the per-token KL between the base model and its fp8 version is NOT uniform across contexts. The
likelihood-ratio test needs about 87 000 tokens at the average KL (about 0.003 nat per token, measured on
the stored ref replies), but a context family whose KL is 10 times larger would need 10 times fewer tokens.
The local pair (base, fp8 simulation) can rank prompts by KL BEFORE any request: sample replies from the
base model, and measure the mean per-token log-ratio log p_base - log p_fp8 on them.

CRITERION FROZEN BEFORE THE RUN (first look at the cells will be this script's output):
* Over the cells (prompts) common to the 9 ref that were already scored by `d_local_run` (120 cells), the
  Spearman correlation between the LOCAL per-token KL of the prompt (variant `w_tensor`, `SAMPLES`
  replies sampled from the base at temperature 1, at most `MAX_NEW_TOKENS` tokens) and the OBSERVED
  per-token likelihood-ratio difference ref-02 minus ref-01 on that cell is >= 0.3 with p <= 0.01.
  If it holds, the local KL predicts where fp8 is visible and prompts can be mined for it; if not, the
  idea is abandoned.
* Reported either way: the share of the total KL carried by the top 10 % of the cells (concentration).
"""

from __future__ import annotations

from typing import Any

import torch

SAMPLES = 4
MAX_NEW_TOKENS = 96
SAMPLING_SEED = 20261121
MIN_SPEARMAN = 0.3
ALPHA = 0.01
TOP_SHARE = 0.10


@torch.no_grad()
def local_kl_per_token(base: Any, quantised: Any, tokenizer: Any, prompt: str, samples: int = SAMPLES,
                       max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[float, int]:
    """(mean log p_base - log p_quantised over sampled tokens, number of tokens) for one prompt.

    Replies are SAMPLED from the base model at temperature 1 (the distribution the plain endpoint draws
    from), so the mean log-ratio estimates KL(base || quantised) per token on this context.
    """
    context = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids
    torch.manual_seed(SAMPLING_SEED)
    batch = base.generate(
        context_ids, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new_tokens,
        num_return_sequences=samples, pad_token_id=tokenizer.eos_token_id)
    total, count = 0.0, 0
    for sequence in batch:
        generated = sequence[context_ids.shape[1]:]
        ends = (generated == tokenizer.eos_token_id).nonzero()
        if len(ends):
            generated = generated[: int(ends[0, 0]) + 1]
        if generated.shape[0] == 0:
            continue
        ids = torch.cat([context_ids[0], generated])[None]
        base_lp = torch.log_softmax(base(ids).logits[0, context_ids.shape[1] - 1:-1], dim=-1)
        quant_lp = torch.log_softmax(quantised(ids).logits[0, context_ids.shape[1] - 1:-1], dim=-1)
        picked = generated[:, None]
        total += float((base_lp.gather(1, picked) - quant_lp.gather(1, picked)).sum())
        count += int(generated.shape[0])
    return (total / count if count else 0.0), count
