"""Log-likelihood scoring of team E on a tiny random Llama (CPU, no weights, no network)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hackathon.model_equality.research import e_loglik as e
from hackathon.model_equality.research.e_models import CANDIDATES, get_candidate
from hackathon.model_equality.store import Answer

VOCAB = 48
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.train(False)
    return model


def _pairs() -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(3)
    pairs = []
    for context_len, reply_len in [(5, 7), (9, 3), (4, 0), (6, 12), (5, 1)]:
        pairs.append((rng.integers(1, VOCAB, context_len).tolist(), rng.integers(1, VOCAB, reply_len).tolist()))
    return pairs


def _reference_scores(
    model: LlamaForCausalLM, pairs: list[tuple[list[int], list[int]]]
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Brute force: one full forward per row, plain log_softmax."""
    out = []
    for context, reply in pairs:
        if not reply:
            z = np.zeros(0, dtype=np.float32)
            out.append((z, z, z))
            continue
        ids = torch.tensor([context + reply])
        with torch.no_grad():
            logits = model(ids).logits[0, len(context) - 1 : -1].float()
        logp_all = torch.log_softmax(logits, dim=-1)
        p = logp_all.exp()
        entropy = -(p * logp_all).sum(-1)
        var = (p * (logp_all + entropy[:, None]).square()).sum(-1)
        chosen = logp_all[torch.arange(len(reply)), torch.tensor(reply)]
        out.append((chosen.numpy(), entropy.numpy(), var.numpy()))
    return out


def test_make_batches_covers_every_row_once_and_respects_budget() -> None:
    lengths = [10, 50, 30, 5, 5, 40, 1]
    batches = e.make_batches(lengths, 100, single_sequence=False)
    flat = sorted(i for batch in batches for i in batch)
    assert flat == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) * lengths[batch[0]] <= 100 or len(batch) == 1
    singles = e.make_batches(lengths, 10_000, single_sequence=True)
    assert all(len(batch) == 1 for batch in singles)


def test_make_batches_empty_input() -> None:
    assert e.make_batches([], 10, single_sequence=False) == []


def test_position_statistics_uniform_and_peaked() -> None:
    logits = torch.zeros((1, VOCAB))
    chosen, entropy, variance = e.position_statistics(logits, torch.tensor([3]))
    assert math.isclose(float(chosen), -math.log(VOCAB), rel_tol=1e-5)
    assert math.isclose(float(entropy), math.log(VOCAB), rel_tol=1e-5)
    assert float(variance) == pytest.approx(0.0, abs=1e-5)
    peaked = torch.full((1, VOCAB), -50.0)
    peaked[0, 7] = 50.0
    chosen, entropy, variance = e.position_statistics(peaked, torch.tensor([7]))
    assert float(chosen) == pytest.approx(0.0, abs=1e-5)
    assert float(entropy) == pytest.approx(0.0, abs=1e-5)


def test_position_statistics_variance_matches_definition() -> None:
    torch.manual_seed(1)
    logits = torch.randn((6, VOCAB)) * 2.0
    targets = torch.randint(0, VOCAB, (6,))
    _, entropy, variance = e.position_statistics(logits, targets)
    log_p = torch.log_softmax(logits, dim=-1)
    manual = (log_p.exp() * (log_p + entropy[:, None]).square()).sum(-1)
    assert torch.allclose(variance, manual, atol=1e-4)


def test_score_pairs_matches_brute_force_batched_and_single(tiny_model: LlamaForCausalLM) -> None:
    pairs = _pairs()
    reference = _reference_scores(tiny_model, pairs)
    batched = e.score_pairs(tiny_model, pairs, CPU, max_batch_tokens=64)
    single = e.score_pairs(tiny_model, pairs, CPU, single_sequence=True)
    assert batched.n_tokens.tolist() == [7, 3, 0, 12, 1]
    for scores in (batched, single):
        offsets = scores.offsets()
        for row, (logp, ent, var) in enumerate(reference):
            span = slice(offsets[row], offsets[row + 1])
            np.testing.assert_allclose(scores.logp[span], logp, atol=1e-4)
            np.testing.assert_allclose(scores.ent[span], ent, atol=1e-4)
            np.testing.assert_allclose(scores.var[span], var, atol=1e-3)


def test_score_pairs_all_empty_replies(tiny_model: LlamaForCausalLM) -> None:
    scores = e.score_pairs(tiny_model, [([1, 2, 3], []), ([4, 5], [])], CPU)
    assert scores.n_tokens.tolist() == [0, 0]
    assert scores.logp.size == 0


def test_calibration_z_is_standard_normal_for_samples_of_the_model(tiny_model: LlamaForCausalLM) -> None:
    """Replies sampled at temperature 1 from the model give z near 0; greedy replies give z > 0."""
    generator = torch.Generator().manual_seed(11)
    sampled_pairs: list[tuple[list[int], list[int]]] = []
    greedy_pairs: list[tuple[list[int], list[int]]] = []
    for _ in range(40):
        context = torch.randint(1, VOCAB, (6,), generator=generator).tolist()
        sampled, greedy = list(context), list(context)
        for _ in range(30):
            with torch.no_grad():
                logits_s = tiny_model(torch.tensor([sampled])).logits[0, -1]
                logits_g = tiny_model(torch.tensor([greedy])).logits[0, -1]
            sampled.append(int(torch.multinomial(torch.softmax(logits_s, -1), 1, generator=generator)))
            greedy.append(int(logits_g.argmax()))
        sampled_pairs.append((context, sampled[6:]))
        greedy_pairs.append((context, greedy[6:]))
    z_sampled = e.calibration_z(*_flat(e.score_pairs(tiny_model, sampled_pairs, CPU)))
    z_greedy = e.calibration_z(*_flat(e.score_pairs(tiny_model, greedy_pairs, CPU)))
    assert abs(z_sampled) < 3.5
    assert z_greedy > 3.5


def _flat(scores: e.RowScores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return scores.logp, scores.ent, scores.var


def test_calibration_z_zero_variance_is_zero() -> None:
    zeros = np.zeros(4, dtype=np.float32)
    assert e.calibration_z(zeros, zeros, zeros) == 0.0


def test_quantise_e4m3_relative_error_and_clamp() -> None:
    values = torch.linspace(-3.0, 3.0, 201)
    scale = values.abs().max() / e.E4M3_MAX
    quantised = e.quantise_e4m3(values, scale)
    big = values.abs() > 0.1
    relative = ((quantised - values).abs() / values.abs())[big]
    assert float(relative.max()) <= 2.0**-4 + 1e-6
    clipped = e.quantise_e4m3(torch.tensor([1000.0]), torch.tensor(1.0))
    assert float(clipped) == e.E4M3_MAX


def test_fp8_target_linears_excludes_lm_head_and_embeddings(tiny_model: LlamaForCausalLM) -> None:
    targets = e.fp8_target_linears(tiny_model)
    assert len(targets) == 2 * 7  # two layers, q k v o gate up down
    assert all(name.startswith("layers.") for name in targets)


@pytest.mark.parametrize("variant", e.FP8_VARIANTS)
def test_fp8_simulator_changes_outputs_and_restores_exactly(
    variant: str, tiny_model: LlamaForCausalLM
) -> None:
    pairs = _pairs()
    base = e.score_pairs(tiny_model, pairs, CPU, single_sequence=True)
    before = {k: v.weight.detach().clone() for k, v in e.fp8_target_linears(tiny_model).items()}
    simulator = e.Fp8Simulator(tiny_model)
    simulator.apply(variant)
    quantised = e.score_pairs(tiny_model, pairs, CPU, single_sequence=True)
    assert not np.allclose(base.logp, quantised.logp, atol=1e-6)
    assert float(np.abs(base.logp - quantised.logp).max()) < 1.0  # a small perturbation, not noise
    simulator.apply(e.VARIANT_BASE)
    restored = e.score_pairs(tiny_model, pairs, CPU, single_sequence=True)
    np.testing.assert_array_equal(base.logp, restored.logp)
    for name, layer in e.fp8_target_linears(tiny_model).items():
        assert torch.equal(layer.weight, before[name])
    assert simulator._hooks == []


def test_fp8_variants_do_not_compound(tiny_model: LlamaForCausalLM) -> None:
    simulator = e.Fp8Simulator(tiny_model)
    simulator.apply(e.VARIANT_W_TENSOR)
    first = e.fp8_target_linears(tiny_model)["layers.0.mlp.up_proj"].weight.detach().clone()
    simulator.apply(e.VARIANT_W_CHANNEL)
    simulator.apply(e.VARIANT_W_TENSOR)
    again = e.fp8_target_linears(tiny_model)["layers.0.mlp.up_proj"].weight.detach().clone()
    assert torch.equal(first, again)
    simulator.apply(e.VARIANT_BASE)


def test_fp8_simulator_rejects_unknown_variant(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(e.LoglikError, match="unknown fp8 variant"):
        e.Fp8Simulator(tiny_model).apply("fp4")


def test_save_and_load_scores_round_trip(tmp_path: Path, tiny_model: LlamaForCausalLM) -> None:
    pairs = _pairs()
    answers = [Answer("ep-01", f"p{i}", 0, "prompt", "reply") for i in range(len(pairs))]
    scores = e.score_pairs(tiny_model, pairs, CPU)
    path = tmp_path / "out" / "ll.npz"
    e.save_scores(path, answers, {("default", "base"): scores}, {"candidate": "toy"})
    assert not path.with_name("ll.npz.tmp").exists()
    header, loaded = e.load_scores(path)
    assert header["endpoint"] == ["ep-01"] * len(pairs)
    assert header["meta"] == {"candidate": "toy"}
    result = loaded[("default", "base")]
    np.testing.assert_array_equal(result.logp, scores.logp)
    np.testing.assert_array_equal(result.n_tokens, scores.n_tokens)


def test_render_context_nosys_per_kind() -> None:
    assert e.render_context(None, "qwen", "nosys", "hi") == (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert e.render_context(None, "smollm", "nosys", "hi").startswith("<|im_start|>user")
    llama = e.render_context(None, "llama3", "nosys", "hi")
    assert llama.startswith("<|begin_of_text|><|start_header_id|>user")
    assert llama.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "system" not in llama
    dated = e.render_context(None, "llama3", "dated", "hi")
    assert "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>" in dated
    assert dated.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
    with pytest.raises(e.LoglikError):
        e.render_context(None, "qwen", "dated", "hi")
    with pytest.raises(e.LoglikError):
        e.render_context(None, "qwen", "weird", "hi")


def test_render_context_default_uses_the_chat_template() -> None:
    seen: dict[str, Any] = {}

    def apply_chat_template(messages: Any, **kwargs: Any) -> str:
        seen["messages"], seen["kwargs"] = messages, kwargs
        return "<templated>"

    tokenizer = SimpleNamespace(apply_chat_template=apply_chat_template)
    assert e.render_context(tokenizer, "qwen", "default", "hello") == "<templated>"
    assert seen["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["kwargs"] == {"add_generation_prompt": True, "tokenize": False}


def test_tokenise_rows_caches_contexts_and_tokenises_reply_alone() -> None:
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool) -> Any:
            calls.append((text, add_special_tokens))
            return SimpleNamespace(input_ids=[ord(c) % 40 for c in text])

        def apply_chat_template(self, messages: Any, **_kwargs: Any) -> str:
            return "CTX:" + str(messages[0]["content"])

    answers = [Answer("ep-01", "a", 0, "same", "x"), Answer("ep-02", "a", 0, "same", "yz")]
    pairs = e.tokenise_rows(FakeTokenizer(), "qwen", "default", answers)
    assert pairs[0][0] == pairs[1][0]
    assert [len(reply) for _, reply in pairs] == [1, 2]
    assert all(flag is False for _, flag in calls)
    assert sum(1 for text, _ in calls if text.startswith("CTX:")) == 1


def test_plan_assignments_balances_and_orders_stages() -> None:
    specs = list(CANDIDATES)
    buckets = e.plan_assignments(specs, 4)
    flat = [(t.candidate, t.stage) for bucket in buckets for t in bucket]
    n_core = sum(1 for spec in specs if spec.tier == "core")
    assert len(flat) == len(set(flat)) == len(specs) + n_core
    assert not any(stage == e.STAGE_FP8 and get_candidate(key).tier == "extra" for key, stage in flat)
    for bucket in buckets:
        stages = [t.stage for t in bucket]
        assert stages == sorted(stages, key=e.STAGES.index)
    loads = [sum(t.cost for t in bucket) for bucket in buckets]
    assert max(loads) <= 1.6 * (sum(loads) / 4) + max(t.cost for b in buckets for t in b)
    with pytest.raises(e.LoglikError):
        e.plan_assignments(specs, 0)


def test_stage_policies() -> None:
    assert e.dtype_for(get_candidate("qwen2.5-14b"), e.STAGE_FP8) == e.DTYPE_BF16
    assert e.dtype_for(get_candidate("llama3.1-8b"), e.STAGE_FP8) == e.DTYPE_FP32
    assert e.dtype_for(get_candidate("llama3.1-8b"), e.STAGE_MATRIX) == e.DTYPE_BF16
    assert e.rows_for(get_candidate("smollm2-135m"), e.STAGE_FP8) == e.ROWS_ALL
    assert e.rows_for(get_candidate("qwen2.5-7b"), e.STAGE_FP8) == e.ROWS_EP
    assert e.rows_for(get_candidate("qwen2.5-7b"), e.STAGE_MATRIX) == e.ROWS_ALL
    assert e.variants_for(e.STAGE_MATRIX) == (e.VARIANT_BASE,)
    assert e.variants_for(e.STAGE_FP8)[0] == e.VARIANT_BASE
    assert e.templates_for(e.STAGE_FP8) == (e.TEMPLATE_DEFAULT,)
    assert e.templates_for(e.STAGE_MATRIX, get_candidate("llama3-8b")) == e.TEMPLATE_VARIANTS
    assert e.templates_for(e.STAGE_MATRIX, get_candidate("llama3.1-8b")) == (
        *e.TEMPLATE_VARIANTS, e.TEMPLATE_DATED)
    assert e.templates_for(e.STAGE_MATRIX, get_candidate("olmo2-7b")) == (e.TEMPLATE_DEFAULT,)


def test_load_answers_filters_and_dedups(tmp_path: Path) -> None:
    def line(ep: str, pid: str, rep: int, resp: str) -> str:
        return (
            f'{{"endpoint": "{ep}", "id": "{pid}", "repeat": {rep}, '
            f'"prompt": "p", "response": "{resp}"}}\n'
        )

    (tmp_path / "answers.jsonl").write_text(line("ep-01", "b", 0, "x") + line("ep-01", "a", 0, "y"))
    (tmp_path / "reference_answers.jsonl").write_text(line("ref-01", "a", 0, "z") + line("ref-01", "a", 0, "dup"))
    everything = e.load_answers(tmp_path)
    assert [(a.endpoint, a.prompt_id) for a in everything] == [("ep-01", "a"), ("ep-01", "b"), ("ref-01", "a")]
    assert [a.response for a in everything if a.endpoint == "ref-01"] == ["z"]
    assert {a.endpoint for a in e.load_answers(tmp_path, rows="ep")} == {"ep-01"}
    assert {a.endpoint for a in e.load_answers(tmp_path, rows="ref")} == {"ref-01"}
    with pytest.raises(e.LoglikError):
        e.load_answers(tmp_path, rows="everything")

def test_tempered_statistics_match_scaled_logits_and_temperature_one(
    tiny_model: LlamaForCausalLM,
) -> None:
    pairs = _pairs()
    grid = (0.5, 1.0, 2.0)
    scores = e.score_pairs(tiny_model, pairs, CPU, temperatures=grid, max_batch_tokens=64)
    plain = e.score_pairs(tiny_model, pairs, CPU)
    assert scores.temperatures == grid
    np.testing.assert_allclose(scores.logp, plain.logp, atol=1e-5)
    assert scores.t_logp is not None and scores.t_ent is not None and scores.t_var is not None
    assert scores.t_logp.shape == (3, int(scores.n_tokens.sum()))
    # temperature 1.0 of the grid reproduces the base statistics
    np.testing.assert_allclose(scores.t_logp[1], scores.logp, atol=1e-5)
    np.testing.assert_allclose(scores.t_ent[1], scores.ent, atol=1e-5)
    # brute force at T = 0.5 on the first row with a reply
    context, reply = pairs[0]
    ids = torch.tensor([context + reply])
    with torch.no_grad():
        logits = tiny_model(ids).logits[0, len(context) - 1 : -1].float() / 0.5
    expected = torch.log_softmax(logits, dim=-1)[torch.arange(len(reply)), torch.tensor(reply)]
    np.testing.assert_allclose(scores.t_logp[0][: len(reply)], expected.numpy(), atol=1e-4)
    # a colder distribution is lower-entropy
    assert float(scores.t_ent[0].mean()) < float(scores.ent.mean()) < float(scores.t_ent[2].mean())


def test_calibration_z_at_the_sampling_temperature_is_near_zero(tiny_model: LlamaForCausalLM) -> None:
    """Replies sampled at T = 0.7 look over-probable at T = 1 but calibrated at T = 0.7."""
    generator = torch.Generator().manual_seed(5)
    sampling_temperature = 0.7
    pairs: list[tuple[list[int], list[int]]] = []
    for _ in range(40):
        context = torch.randint(1, VOCAB, (6,), generator=generator).tolist()
        sequence = list(context)
        for _ in range(30):
            with torch.no_grad():
                logits = tiny_model(torch.tensor([sequence])).logits[0, -1] / sampling_temperature
            sequence.append(int(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)))
        pairs.append((context, sequence[6:]))
    scores = e.score_pairs(tiny_model, pairs, CPU, temperatures=(sampling_temperature,))
    assert scores.t_logp is not None and scores.t_ent is not None and scores.t_var is not None
    z_at_one = e.calibration_z(scores.logp, scores.ent, scores.var)
    z_at_t = e.calibration_z(scores.t_logp[0], scores.t_ent[0], scores.t_var[0])
    assert abs(z_at_t) < 3.5
    assert z_at_one > abs(z_at_t)


def test_save_and_load_round_trip_keeps_tempered_arrays(tmp_path: Path, tiny_model: LlamaForCausalLM) -> None:
    pairs = _pairs()
    answers = [Answer("ep-01", f"p{i}", 0, "prompt", "reply") for i in range(len(pairs))]
    scores = e.score_pairs(tiny_model, pairs, CPU, temperatures=(0.5, 1.5))
    path = tmp_path / "t.npz"
    e.save_scores(path, answers, {("default", "base"): scores}, {})
    _, loaded = e.load_scores(path)
    result = loaded[("default", "base")]
    assert result.temperatures == (0.5, 1.5)
    assert result.t_logp is not None and scores.t_logp is not None
    np.testing.assert_array_equal(result.t_logp, scores.t_logp)


@pytest.mark.parametrize("mode", [e.WEIGHT_TENSOR, e.WEIGHT_CHANNEL, e.WEIGHT_BLOCK])
def test_quantise_weight_shape_and_relative_error(mode: str) -> None:
    torch.manual_seed(2)
    weight = torch.randn(300, 200) * 0.05  # not a multiple of the 128 block on either side
    quantised = e.quantise_weight(weight, mode)
    assert quantised.shape == weight.shape
    big = weight.abs() > 0.02
    relative = ((quantised - weight).abs() / weight.abs())[big]
    assert float(relative.mean()) < 2.0**-4
    assert not torch.equal(quantised, weight)


def test_block_scales_follow_local_amplitude() -> None:
    """E4M3 underflows tiny values next to a huge one under a shared scale; block scales avoid it."""
    torch.manual_seed(4)
    weight = torch.randn(256, 256) * 1e-5
    weight[:128, :128] = torch.randn(128, 128) * 5.0
    small = (slice(128, 256), slice(128, 256))
    error_tensor = (e.quantise_weight(weight, e.WEIGHT_TENSOR)[small] - weight[small]).abs().mean()
    error_block = (e.quantise_weight(weight, e.WEIGHT_BLOCK)[small] - weight[small]).abs().mean()
    assert float(error_block) < float(error_tensor)


def test_quantise_weight_rejects_unknown_mode() -> None:
    with pytest.raises(e.LoglikError):
        e.quantise_weight(torch.zeros(4, 4), "ternary")


def test_new_fp8_stages_policies() -> None:
    spec = get_candidate("qwen2.5-3b")
    assert e.dtype_for(spec, e.STAGE_FP8BF16) == e.DTYPE_BF16
    assert e.dtype_for(spec, e.STAGE_FP8X) == e.DTYPE_FP32
    assert e.variants_for(e.STAGE_FP8X) == (e.VARIANT_BASE, *e.FP8_VARIANTS_BLOCK)
    assert e.variants_for(e.STAGE_FP8BF16)[0] == e.VARIANT_BASE
    assert e.rows_for(get_candidate("smollm2-360m"), e.STAGE_FP8BF16) == e.ROWS_ALL
    assert e.rows_for(spec, e.STAGE_FP8X) == e.ROWS_EP
    buckets = e.plan_assignments([spec, get_candidate("smollm2-360m")], 2, (e.STAGE_FP8X, e.STAGE_FP8BF16))
    flat = {(t.candidate, t.stage) for bucket in buckets for t in bucket}
    assert len(flat) == 4


def test_fp8lite_stage_is_fp32_with_two_recipes() -> None:
    spec = get_candidate("qwen2.5-3b")
    assert e.STAGE_FP8LITE in e.FP8_STAGES and e.STAGE_FP8LITE in e.ALL_STAGES
    assert e.dtype_for(spec, e.STAGE_FP8LITE) == e.DTYPE_FP32
    assert e.dtype_for(get_candidate("qwen2.5-14b"), e.STAGE_FP8LITE) == e.DTYPE_BF16
    assert e.variants_for(e.STAGE_FP8LITE) == (e.VARIANT_BASE, e.VARIANT_W_TENSOR, e.VARIANT_W_CHANNEL)
    assert e.templates_for(e.STAGE_FP8LITE, spec) == (e.TEMPLATE_DEFAULT,)
    assert e.rows_for(spec, e.STAGE_FP8LITE) == e.ROWS_EP


def test_parse_task_list_and_errors() -> None:
    tasks = e.parse_task_list(["qwen2.5-3b:matrix", "llama3.2-1b:fp8lite"])
    assert [(t.candidate, t.stage) for t in tasks] == [("qwen2.5-3b", "matrix"), ("llama3.2-1b", "fp8lite")]
    for bad in ("qwen2.5-3b", "qwen2.5-3b:nope", "nobody:matrix"):
        with pytest.raises(e.LoglikError):
            e.parse_task_list([bad])


@pytest.mark.parametrize("gpu_count", [1, 2, 3, 4, 6])
def test_distribute_groups_keeps_groups_whole_and_covers_every_task(gpu_count: int) -> None:
    groups = [e.parse_task_list([f"{c}:matrix", f"{c}:fp8lite"]) for c in ("qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b")]
    extras = e.parse_task_list([f"{c}:matrix" for c in ("qwen2.5-7b", "llama3.2-3b", "qwen2.5-14b", "qwen3-8b")])
    buckets = e.distribute_groups(groups, extras, gpu_count)
    assert len(buckets) == gpu_count
    flat = [(t.candidate, t.stage) for bucket in buckets for t in bucket]
    assert len(flat) == len(set(flat)) == 6 + 4
    for group in groups:  # a base model's two stages stay together and in order on one GPU
        homes = [i for i, bucket in enumerate(buckets) if group[0] in bucket]
        assert len(homes) == 1
        bucket = buckets[homes[0]]
        assert bucket.index(group[1]) == bucket.index(group[0]) + 1
    if gpu_count >= 4:  # the spare GPU gets work too
        assert len(buckets[3]) > 0
    with pytest.raises(e.LoglikError):
        e.distribute_groups(groups, extras, 0)


def test_write_container_retries_on_a_full_quota_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    calls = {"n": 0}
    real_savez = np.savez

    def flaky_savez(handle: Any, **arrays: Any) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        real_savez(handle, **arrays)

    monkeypatch.setattr(np, "savez", flaky_savez)
    sleeps: list[float] = []
    target = tmp_path / "out.npz"
    e.write_container(target, {"x": np.arange(3)}, attempts=5, delay=15.0, sleep=sleeps.append)
    assert calls["n"] == 3 and sleeps == [15.0, 15.0]
    with np.load(target) as data:
        assert data["x"].tolist() == [0, 1, 2]
    assert not (tmp_path / "out.npz.tmp").exists()


def test_write_container_gives_up_with_a_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    def always_full(handle: Any, **arrays: Any) -> None:
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(np, "savez", always_full)
    with pytest.raises(e.LoglikError, match="after 3 attempts"):
        e.write_container(tmp_path / "out.npz", {"x": np.zeros(1)}, attempts=3, delay=0.0, sleep=lambda _: None)


def test_write_container_fills_a_placeholder_in_place(tmp_path: Path) -> None:
    target = tmp_path / "placeholder.npz"
    target.write_bytes(b"")
    inode_before = target.stat().st_ino
    e.write_container(target, {"x": np.arange(4)}, attempts=1)
    assert target.stat().st_ino == inode_before  # no new inode: the existing file was written into
    with np.load(target) as data:
        assert data["x"].tolist() == [0, 1, 2, 3]


def test_fallback_line_carries_the_per_row_sums(tiny_model: LlamaForCausalLM) -> None:
    pairs = _pairs()
    answers = [Answer("ep-01", f"p{i}", 0, "prompt", "reply") for i in range(len(pairs))]
    scores = e.score_pairs(tiny_model, pairs, CPU, temperatures=(0.5, 1.5))
    line = e.fallback_line("toy", "matrix", answers, {("default", "base"): scores})
    assert line.startswith(e.FALLBACK_PREFIX + " ")
    payload = json.loads(line[len(e.FALLBACK_PREFIX) + 1 :])
    entry = payload["variants"]["default/base"]
    assert payload["rows"][0] == ["ep-01", "p0", 0] and len(payload["rows"]) == len(pairs)
    assert entry["n_tokens"] == [7, 3, 0, 12, 1]
    offsets = scores.offsets()
    assert entry["logp"][0] == pytest.approx(float(scores.logp[offsets[0] : offsets[1]].sum()), abs=1e-4)
    assert entry["logp"][2] == 0.0  # the empty row
    assert len(entry["t_logp"]) == 2 and len(entry["t_logp"][0]) == len(pairs)
    assert entry["temperatures"] == [0.5, 1.5]


def test_two_gpus_put_the_3b_base_alone_and_the_two_small_bases_together() -> None:
    groups = [e.parse_task_list([f"{c}:matrix", f"{c}:fp8lite"]) for c in ("qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b")]
    extras = e.parse_task_list([f"{c}:matrix" for c in ("qwen2.5-7b", "llama3.2-3b", "qwen2.5-14b")])
    buckets = e.distribute_groups(groups, extras, 2)
    names = [[(t.candidate, t.stage) for t in bucket] for bucket in buckets]
    assert names[0][:2] == [("qwen2.5-3b", "matrix"), ("qwen2.5-3b", "fp8lite")]
    assert ("llama3.2-1b", "matrix") not in names[0][:4]
    assert names[1][:4] == [
        ("qwen2.5-1.5b", "matrix"), ("qwen2.5-1.5b", "fp8lite"),
        ("llama3.2-1b", "matrix"), ("llama3.2-1b", "fp8lite"),
    ]
    # the three families come before any loose task, on both GPUs
    for bucket in names:
        families = [i for i, (c, _) in enumerate(bucket) if c in ("qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b")]
        loose = [i for i, (c, _) in enumerate(bucket) if c not in ("qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b")]
        assert not families or not loose or max(families) < min(loose)


def test_one_gpu_per_base_with_three_or_more_gpus() -> None:
    groups = [e.parse_task_list([f"{c}:matrix", f"{c}:fp8lite"]) for c in ("qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b")]
    for gpu_count in (3, 4):
        buckets = e.distribute_groups(groups, [], gpu_count)
        assert [bucket[0].candidate for bucket in buckets[:3]] == ["qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b"]


def test_visible_gpu_ids_prefers_the_runner_allocation() -> None:
    assert e.visible_gpu_ids("2,3", 4) == ["2", "3"]
    assert e.visible_gpu_ids(" 1 ", 4) == ["1"]
    assert e.visible_gpu_ids(None, 3) == ["0", "1", "2"]
    assert e.visible_gpu_ids("", 2) == ["0", "1"]


def test_block_variants_apply_and_restore(tiny_model: LlamaForCausalLM) -> None:
    pairs = _pairs()
    base = e.score_pairs(tiny_model, pairs, CPU, single_sequence=True)
    simulator = e.Fp8Simulator(tiny_model)
    for variant in (e.VARIANT_W_BLOCK, e.VARIANT_W_BLOCK_A8_TOKEN):
        simulator.apply(variant)
        changed = e.score_pairs(tiny_model, pairs, CPU, single_sequence=True)
        assert not np.allclose(base.logp, changed.logp, atol=1e-7)
    simulator.apply(e.VARIANT_BASE)
    np.testing.assert_array_equal(base.logp, e.score_pairs(tiny_model, pairs, CPU, single_sequence=True).logp)
