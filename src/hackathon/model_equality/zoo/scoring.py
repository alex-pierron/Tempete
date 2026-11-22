"""Per-token log-likelihood of an answer under a model: the statistic of team D's likelihood flag.

The recipe is the one of `research/d_local.reply_log_probability`, frozen by team D for the ref and
meant to be reused unchanged on the large families: context = chat template with the user message
only (no system prompt, whatever the endpoint had), generation prompt appended; the answer is
tokenised with `add_special_tokens=False`; the statistic is the sum of the answer-token
log-probabilities and the number of answer tokens. Computed here in bf16 on the GPU (team D used
float32 on CPU for the 135M model), with a float32 log-softmax.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from hackathon.model_equality.zoo.sampling import chat_text, padding_id

PairScore = tuple[float, int]
"""`(sum of answer-token log-probabilities, number of answer tokens)`."""


def score_replies(
    model: Any, tokenizer: Any, pairs: Sequence[tuple[str, str]], batch_size: int,
    device: torch.device,
) -> list[PairScore]:
    """Score each `(user message, answer)`; an empty answer scores `(0.0, 0)`.

    Sequences are batched by length (right padding); only the answer positions go through the
    float32 log-softmax, so the memory peak is the bf16 logits of one batch.
    """
    contexts = [tokenizer(chat_text(tokenizer, user), add_special_tokens=False)["input_ids"]
                for user, _ in pairs]
    replies = [tokenizer(reply, add_special_tokens=False)["input_ids"] for _, reply in pairs]
    pad = padding_id(tokenizer, [int(tokenizer.eos_token_id or 0)])
    scores: list[PairScore] = [(0.0, 0)] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: len(contexts[i]) + len(replies[i]))
    model.eval()
    for start in range(0, len(order), batch_size):
        indices = [i for i in order[start:start + batch_size] if replies[i]]
        if not indices:
            continue
        sequences = [contexts[i] + replies[i] for i in indices]
        longest = max(len(s) for s in sequences)
        ids = torch.full((len(indices), longest), pad, dtype=torch.long)
        mask = torch.zeros((len(indices), longest), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[row, :len(sequence)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            for row, item in enumerate(indices):
                first = len(contexts[item])
                count = len(replies[item])
                window = logits[row, first - 1:first + count - 1].to(torch.float32)
                log_probabilities = torch.log_softmax(window, dim=-1)
                targets = torch.tensor(replies[item], dtype=torch.long, device=device)
                total = log_probabilities.gather(1, targets[:, None]).sum()
                scores[item] = (float(total), count)
    return scores
