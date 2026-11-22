"""Zoo (team F): config, prompts, fp8, LoRA, sampling, KGW, scoring, store and a full tiny run.

All on a two-layer random Llama and an in-memory tokenizer (see `test_hk_met_zoo_support`): no
network, no cache, a few seconds on a CPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from test_hk_met_zoo_support import (
    build_model,
    build_tokenizer,
    save_tiny_family,
    write_prompt_files,
)
from torch import nn

from hackathon.model_equality.zoo import lora as zoo_lora
from hackathon.model_equality.zoo import quant, store, worker
from hackathon.model_equality.zoo.calibrate import (
    aggregate,
    invert_curve,
    round_to_step,
)
from hackathon.model_equality.zoo.config import (
    BASE_SLOTS,
    FAMILIES,
    GENERATION_ORDER,
    IDENTITY_DRAWS,
    REPEATS_ARITHMETIC,
    VARIANT_BY_KEY,
    VARIANTS,
    ZooConfigError,
    endpoint_name,
    parse_endpoint,
    variant_for_slot,
)
from hackathon.model_equality.zoo.prompts import (
    PERSONAS,
    SYSTEM_PROMPT,
    PromptBundleError,
    Unit,
    build_bundle,
    bundle_from_json,
    generated_arithmetic,
    load_bundle,
    persona_for,
    stable_index,
)
from hackathon.model_equality.zoo.sampling import chat_text, generate_texts, load_kgw_processor
from hackathon.model_equality.zoo.scoring import score_replies

CPU = torch.device("cpu")
EXTRA = 2


# --- config ---------------------------------------------------------------------------------
def test_variant_table_matches_the_ref_layout() -> None:
    truth = {v.slot: v.truth_model for v in VARIANTS}
    assert sorted(truth) == list(range(1, 11))
    assert {truth[s] for s in (1, 3, 5, 6, 8)} == {"S"}
    assert truth[2] == truth[10] == "T1_fp8"
    assert len({truth[s] for s in BASE_SLOTS}) == 5  # S, fp8, three LoRA strengths
    assert [VARIANT_BY_KEY[k].lora_scale for k in ("lora03", "lora07", "lora15")] == [0.3, 0.7, 1.5]
    assert set(GENERATION_ORDER) == set(VARIANT_BY_KEY)
    assert GENERATION_ORDER.index("fp8_w") > GENERATION_ORDER.index("lora15")


def test_endpoint_names_round_trip_and_reject_garbage() -> None:
    for family in FAMILIES:
        for variant in VARIANTS:
            assert parse_endpoint(endpoint_name(family.key, variant.slot)) == (
                family.key, variant.slot)
    for bad in ("ep-01", "zoo-nope-01", "zoo-qwen15-xx", "zoo-qwen"):
        with pytest.raises(ZooConfigError):
            parse_endpoint(bad)
    with pytest.raises(ZooConfigError):
        endpoint_name("nope", 1)
    with pytest.raises(ZooConfigError):
        variant_for_slot(99)


def test_families_are_the_ep_base_models_plus_the_ref_control() -> None:
    assert [f.key for f in FAMILIES] == ["qwen15", "qwen3", "llama1", "smollm"]
    assert [f.key for f in FAMILIES if f.is_control] == ["smollm"]
    assert all("-" not in f.key for f in FAMILIES)  # endpoint ids split on the dash
    assert "360M" in FAMILIES[-1].display


# --- prompts --------------------------------------------------------------------------------
def test_bundle_units_cover_every_prompt_with_its_repeats(tmp_path: Path) -> None:
    bundle = build_bundle(*write_prompt_files(tmp_path), extra=EXTRA)
    units = bundle.units()
    assert len(bundle.arithmetic) == 5 + EXTRA and len(bundle.identity) == 2
    assert len(units) == (5 + EXTRA) * REPEATS_ARITHMETIC + 2 * IDENTITY_DRAWS
    assert len({(u.prompt_id, u.repeat) for u in units}) == len(units)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle.to_json()), encoding="utf-8")
    assert load_bundle(path) == bundle


def test_generated_arithmetic_is_deterministic_unique_and_in_the_ref_template() -> None:
    taken = {"Compute 1 + 2. Answer with only the number."}
    first = generated_arithmetic(30, 5, taken)
    assert first == generated_arithmetic(30, 5, taken)
    assert first != generated_arithmetic(30, 6, taken)
    assert len({row["prompt"] for row in first}) == 30
    assert [row["id"] for row in first][:2] == ["z-arith-000", "z-arith-001"]
    pattern = re.compile(r"^Compute (\d+) ([+*-]) (\d+)\. Answer with only the number\.$")
    for row in first:
        match = pattern.match(row["prompt"])
        assert match is not None
        left, operator, right = int(match[1]), match[2], int(match[3])
        if operator == "-":
            assert right <= left  # the ref differences are positive
        if operator == "*":
            assert 10 <= left <= 99 and 100 <= right <= 999
    assert {row["prompt"][8:].split()[1] for row in first} == {"+", "-", "*"}


def test_system_prompt_is_short_and_generic_and_personas_are_roles() -> None:
    assert SYSTEM_PROMPT == "You are a helpful assistant."
    assert all(persona.lstrip("*").startswith("You are ") and "specialized in" in persona
               for persona in PERSONAS)
    assert len(set(PERSONAS)) == len(PERSONAS) == 11


def test_bundle_rejects_duplicates_and_bad_versions() -> None:
    with pytest.raises(PromptBundleError):
        bundle_from_json({"version": 1, "arithmetic": [{"id": "a", "prompt": "x"},
                                                       {"id": "a", "prompt": "y"}],
                          "identity": []})
    with pytest.raises(PromptBundleError):
        bundle_from_json({"version": 99, "arithmetic": [], "identity": []})
    with pytest.raises(PromptBundleError):
        bundle_from_json({"version": 1, "arithmetic": [{"id": "a", "prompt": " "}],
                          "identity": []})


def test_persona_is_stable_and_spreads_over_the_pool() -> None:
    units = [Unit(f"p{k}", r, "x", "arithmetic") for k in range(40) for r in range(2)]
    first = [persona_for("qwen", u) for u in units]
    assert first == [persona_for("qwen", u) for u in units]
    assert len(set(first)) > len(PERSONAS) // 2
    assert [persona_for("llama", u) for u in units] != first
    assert 0 <= stable_index("a", 1, modulo=7) < 7
    with pytest.raises(ZooConfigError):
        stable_index("a", modulo=0)


# --- fp8 ------------------------------------------------------------------------------------
def test_fp8_round_trip_is_close_idempotent_and_on_the_grid() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 64)
    scale = quant.per_tensor_scale(weight)
    once = quant.fake_quantize_e4m3(weight, scale)
    relative = float((once - weight).norm() / weight.norm())
    assert 0.005 < relative < 0.06
    assert torch.equal(quant.fake_quantize_e4m3(once, quant.per_tensor_scale(once)), once)
    assert len(torch.unique(once)) < len(torch.unique(weight))
    assert float(once.abs().max()) == pytest.approx(float(weight.abs().max()), rel=1e-5)


def test_quantize_in_place_reports_error_and_zero_tensor_is_safe() -> None:
    weight = torch.randn(16, 16)
    assert quant.quantize_weight_in_place(weight) > 0.0
    zeros = torch.zeros(4, 4)
    assert quant.quantize_weight_in_place(zeros) == 0.0
    assert torch.equal(zeros, torch.zeros(4, 4))


# --- ZooLinear and LoRA ---------------------------------------------------------------------
def test_zoo_linear_matches_the_base_layer_when_the_adapter_is_off() -> None:
    base = nn.Linear(8, 6)
    layer = zoo_lora.ZooLinear(base)
    layer.attach_lora(4)
    inputs = torch.randn(3, 8)
    assert torch.allclose(layer(inputs), base(inputs))  # scale 0
    layer.lora_scale = 1.0
    assert torch.allclose(layer(inputs), base(inputs))  # B is zero at init


def test_lora_strength_scales_the_delta_linearly() -> None:
    torch.manual_seed(1)
    layer = zoo_lora.ZooLinear(nn.Linear(8, 6))
    layer.attach_lora(4)
    assert layer.lora_a is not None and layer.lora_b is not None
    nn.init.normal_(layer.lora_b)
    inputs = torch.randn(5, 8)
    reference = layer(inputs)
    layer.lora_scale = 1.0
    full = layer(inputs) - reference
    layer.lora_scale = 0.3
    assert torch.allclose(layer(inputs) - reference, 0.3 * full, atol=1e-5)
    layer.lora_scale = 1.5
    assert torch.allclose(layer(inputs) - reference, 1.5 * full, atol=1e-5)


def test_install_wraps_every_projection_without_changing_the_logits() -> None:
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    ids = torch.tensor([[3, 7, 9, 11, 4]])
    before = model(input_ids=ids).logits
    linears = zoo_lora.install_zoo_linears(model)
    assert len(linears) == 2 * 7  # two layers x (q k v o gate up down)
    assert torch.allclose(model(input_ids=ids).logits, before, atol=1e-6)
    with pytest.raises(ZooConfigError):
        zoo_lora.install_zoo_linears(nn.Linear(2, 2))


def test_adapter_round_trip_and_shape_check(tmp_path: Path) -> None:
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    linears = zoo_lora.install_zoo_linears(model)
    zoo_lora.attach_all_lora(linears)
    for layer in linears.values():
        assert layer.lora_b is not None
        nn.init.normal_(layer.lora_b)
    path = tmp_path / "adapter.safetensors"
    zoo_lora.save_adapter(path, linears)
    other = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    other_linears = zoo_lora.install_zoo_linears(other)
    zoo_lora.load_adapter(path, other_linears)
    for name, layer in linears.items():
        assert layer.lora_b is not None and other_linears[name].lora_b is not None
        assert torch.equal(layer.lora_b, other_linears[name].lora_b)  # type: ignore[arg-type]
    bigger = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    bigger.config.hidden_size = 32
    mismatched = {k: zoo_lora.ZooLinear(nn.Linear(5, 5)) for k in linears}
    with pytest.raises(zoo_lora.AdapterError):
        zoo_lora.load_adapter(path, mismatched)


def test_fp8_weights_restore_is_exact_and_leaves_gpu_free_of_copies() -> None:
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    linears = zoo_lora.install_zoo_linears(model)
    originals = {n: layer.weight.detach().clone() for n, layer in linears.items()}
    fp8 = zoo_lora.Fp8Weights(linears)
    error = fp8.quantize()
    assert 0.005 < error < 0.06
    assert any(not torch.equal(linears[n].weight, originals[n]) for n in linears)
    fp8.restore()
    assert all(torch.equal(linears[n].weight, originals[n]) for n in linears)


def test_training_example_labels_cover_the_answer_only() -> None:
    tokenizer = build_tokenizer()
    example = zoo_lora.tokenize_example(tokenizer, "what is your name ?", "i am a assistant")
    assert example is not None
    answered = [i for i, label in enumerate(example.labels) if label != zoo_lora.IGNORE_INDEX]
    assert answered and answered[0] == example.labels.index(next(
        label for label in example.labels if label != zoo_lora.IGNORE_INDEX))
    assert all(example.labels[i] == example.input_ids[i] for i in answered)
    assert tokenizer.convert_tokens_to_ids("<|end|>") in example.input_ids  # learns to stop
    assert zoo_lora.tokenize_example(tokenizer, "what is your name ?", "i am " * 200) is None
    with pytest.raises(zoo_lora.TrainingDataError):
        zoo_lora.render_instruction({"instruction": "", "output": "x"})


def test_learning_rate_schedule_warms_up_then_decays() -> None:
    rates = [zoo_lora.learning_rate_at(s, 100) for s in range(100)]
    assert rates[0] < rates[9] and rates[9] == pytest.approx(zoo_lora.LORA_LEARNING_RATE)
    assert rates[-1] == min(rates[10:]) and rates[-1] >= 0.1 * zoo_lora.LORA_LEARNING_RATE


def test_lora_training_lowers_the_loss_and_leaves_the_base_untouched() -> None:
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    linears = zoo_lora.install_zoo_linears(model)
    base_weights = {n: layer.weight.detach().clone() for n, layer in linears.items()}
    example = zoo_lora.tokenize_example(tokenizer, "what is your name ?", "i am a assistant")
    assert example is not None
    torch.manual_seed(3)
    report = zoo_lora.train_lora(
        model, linears, [example], int(tokenizer.pad_token_id), steps=25, effective_batch=4,
        micro_batch=2, device=CPU, log=lambda _: None, learning_rate=3e-3)
    assert report.losses[-1] < 0.92 * report.losses[0]  # the tiny random model plateaus early
    ids = torch.tensor([example.input_ids])
    off = model(input_ids=ids).logits
    zoo_lora.set_lora_scale(linears, 1.0)
    assert not torch.allclose(model(input_ids=ids).logits, off, atol=1e-4)  # the adapter acts
    zoo_lora.set_lora_scale(linears, 0.0)
    assert all(torch.equal(linears[n].weight, base_weights[n]) for n in linears)
    assert all(layer.lora_scale == 0.0 for layer in linears.values())  # adapter left off
    assert report.parameters > 0 and report.final_micro_batch == 2


def test_training_falls_back_on_out_of_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    linears = zoo_lora.install_zoo_linears(model)
    example = zoo_lora.tokenize_example(tokenizer, "what is your name ?", "i am a assistant")
    assert example is not None
    original = zoo_lora._accumulate_gradients
    seen: list[int] = []

    def flaky(model_: object, batch: object, micro: int, pad: int, device: object) -> float:
        seen.append(micro)
        if micro > 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        return original(model_, batch, micro, pad, device)  # type: ignore[arg-type]

    monkeypatch.setattr(zoo_lora, "_accumulate_gradients", flaky)
    report = zoo_lora.train_lora(
        model, linears, [example], int(tokenizer.pad_token_id), steps=2, effective_batch=8,
        micro_batch=16, device=CPU, log=lambda _: None)
    assert report.oom_events == 2 and report.final_micro_batch == 4
    assert seen[:3] == [16, 8, 4]


# --- sampling, KGW, scoring -----------------------------------------------------------------
def test_chat_text_adds_the_system_message_and_the_generation_prompt() -> None:
    tokenizer = build_tokenizer()
    plain = chat_text(tokenizer, "what is your name ?")
    system = chat_text(tokenizer, "what is your name ?", "you are helpful")
    assert plain.rstrip().endswith("<|assistant|>")
    assert "<|system|> you are helpful" in system and system.endswith(plain[-14:])


def test_generation_is_seeded_ordered_and_bounded() -> None:
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    prompts = [chat_text(tokenizer, text) for text in
               ("what is your name ?", "give your name and creator in one line .", "blue")]
    first = generate_texts(model, tokenizer, prompts, 0.7, False, 2, 11, CPU,
                           max_new_tokens=12)
    again = generate_texts(model, tokenizer, prompts, 0.7, False, 2, 11, CPU,
                           max_new_tokens=12)
    assert first.texts == again.texts and len(first.texts) == 3
    assert max(first.new_tokens) <= 12
    other = generate_texts(model, tokenizer, prompts, 1.0, False, 2, 12, CPU, max_new_tokens=12)
    assert other.texts != first.texts


def test_generation_halves_the_chunk_on_out_of_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from hackathon.model_equality.zoo import sampling

    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    prompts = [chat_text(tokenizer, "blue")] * 8
    original = sampling._generate_chunk
    sizes: list[int] = []

    def flaky(m: object, t: object, chunk: list[str], *rest: object) -> tuple[list[str], list[int]]:
        sizes.append(len(chunk))
        if len(chunk) > 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        return original(m, t, chunk, *rest)  # type: ignore[arg-type]

    monkeypatch.setattr(sampling, "_generate_chunk", flaky)
    result = generate_texts(model, tokenizer, prompts, 0.7, False, 8, 5, CPU, max_new_tokens=4,
                            log=lambda _: None)
    assert sizes[0] == 8 and result.final_batch == 4 and result.oom_events == 1
    assert len(result.texts) == 8


def test_kgw_watermark_is_detected_in_the_generated_tokens() -> None:
    from wmkit.text.kgw import KGWConfig, kgw_detect

    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    assert load_kgw_processor(len(tokenizer)).config.delta == 2.0
    prompts = [chat_text(tokenizer, "give your name and creator in one line .")] * 6
    vocabulary = int(model.config.vocab_size)

    def mean_z(watermark: bool) -> float:
        tokenizer.padding_side = "left"
        result = generate_texts(model, tokenizer, prompts, 1.0, watermark, 6, 21, CPU,
                                max_new_tokens=60)
        scores = []
        for text in result.texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            detection = kgw_detect(ids, vocabulary, KGWConfig())
            scores.append(detection.z_score or 0.0)
        return sum(scores) / len(scores)

    assert mean_z(True) > mean_z(False) + 2.0


def test_score_replies_matches_a_manual_log_probability() -> None:
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    user, reply = "what is your name ?", "i am a assistant"
    (total, count), (empty_total, empty_count) = score_replies(
        model, tokenizer, [(user, reply), (user, "")], 4, CPU)
    context = tokenizer(chat_text(tokenizer, user), add_special_tokens=False)["input_ids"]
    answer = tokenizer(reply, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([context + answer])
    log_probs = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
    manual = sum(float(log_probs[len(context) - 1 + k, answer[k]]) for k in range(len(answer)))
    assert count == len(answer) and total == pytest.approx(manual, abs=1e-4)
    assert (empty_total, empty_count) == (0.0, 0)


# --- store ----------------------------------------------------------------------------------
def test_store_upsert_replaces_an_endpoint_and_survives_a_torn_line(tmp_path: Path) -> None:
    path = tmp_path / "zoo.jsonl"
    rows_a = [store.make_row("zoo-qwen15-01", "p", r, "q", f"a{r}") for r in range(2)]
    rows_b = [store.make_row("zoo-qwen15-02", "p", 0, "q", "b")]
    assert store.commit_endpoint(path, "zoo-qwen15-01", rows_a) == 2
    store.commit_endpoint(path, "zoo-qwen15-02", rows_b)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"endpoint": "zoo-qwen15-0')  # a torn write
    fresh = [store.make_row("zoo-qwen15-01", "p", r, "q", f"new{r}") for r in range(2)]
    store.commit_endpoint(path, "zoo-qwen15-01", fresh)
    counts = store.row_counts(path)
    assert counts == {"zoo-qwen15-01": 2, "zoo-qwen15-02": 1}
    assert {str(r["response"]) for r in store.read_rows(path)
            if r["endpoint"] == "zoo-qwen15-01"} == {"new0", "new1"}
    with pytest.raises(ValueError):
        store.commit_endpoint(path, "zoo-qwen15-03", rows_b)
    assert store.read_rows(tmp_path / "missing.jsonl") == []


def test_store_rows_use_the_official_fields() -> None:
    row = store.make_row("zoo-llama-04", "h-arith-001", 1, "Compute", "42")
    assert set(row) == {"endpoint", "id", "repeat", "prompt", "response", "timestamp"}
    assert str(row["timestamp"]).endswith("Z")


# --- calibration helpers --------------------------------------------------------------------
def test_invert_curve_interpolates_and_flags_out_of_range() -> None:
    temperatures = [0.2, 0.4, 0.6, 0.8]
    curve = [-1.0, -1.4, -2.0, -3.0]
    assert invert_curve(temperatures, curve, -1.7)[0] == pytest.approx(0.5)
    assert invert_curve(temperatures, curve, -0.5) == (0.2, True)
    assert invert_curve(temperatures, curve, -4.0) == (0.8, True)
    noisy = [-1.0, -1.4, -1.3, -3.0]  # not monotone: running minimum applied
    assert invert_curve(temperatures, noisy, -1.4)[1] is False
    assert round_to_step(0.683) == 0.7 and round_to_step(0.62) == 0.6


def test_aggregate_sums_per_prompt_and_means_token_weighted() -> None:
    stats = aggregate(["a", "a", "b"], [(-2.0, 2), (-4.0, 2), (-3.0, 3)])
    assert stats.per_prompt == {"a": (-6.0, 4), "b": (-3.0, 3)}
    assert stats.mean(["a", "b"]) == pytest.approx(-9.0 / 7.0)
    assert stats.mean(["a", "a"]) == pytest.approx(-1.5)


# --- the whole worker on a tiny family ------------------------------------------------------
def _tiny_settings(tmp_path: Path, steps: int = 3) -> worker.RunSettings:
    save_tiny_family(tmp_path / "model")
    bundle = build_bundle(*write_prompt_files(tmp_path), extra=EXTRA)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle.to_json()), encoding="utf-8")
    data = [{"instruction": "what is your name ?", "input": "", "output": "i am a assistant ."}] * 16
    lora_data = tmp_path / "alpaca.json"
    lora_data.write_text(json.dumps(data), encoding="utf-8")
    arguments = worker.build_parser().parse_args([
        "run", "--family", "qwen15", "--bundle", str(bundle_path),
        "--out-dir", str(tmp_path / "out"), "--model-dir", str(tmp_path / "model"),
        "--lora-data", str(lora_data), "--lora-steps", str(steps), "--effective-batch", "4",
        "--device", "cpu", "--dtype", "float32", "--max-new-tokens", "6", "--peers", "qwen15",
        "--peer-wait-minutes", "0.01",
    ])
    return worker.settings_from_args(arguments)


def test_worker_runs_every_variant_scores_and_is_idempotent(tmp_path: Path) -> None:
    settings = _tiny_settings(tmp_path)
    assert worker.run_family(settings) == 0
    units = len(settings.bundle.units())
    counts = store.row_counts(settings.answers_path)
    assert {parse_endpoint(e)[1] for e in counts} == set(range(1, 11))
    assert all(count == units for count in counts.values())
    meta = json.loads(settings.meta_path.read_text(encoding="utf-8"))
    assert meta["stage_a_complete"] and set(meta["variants"]) == set(VARIANT_BY_KEY)
    assert 0.0 < meta["fp8_weight_relative_error"] < 0.1
    assert meta["lora"]["adapter_sha256"] and meta["lora"]["steps"] == 3
    scores = json.loads(settings.likelihood_path.read_text(encoding="utf-8"))["scores"]
    assert len(scores["base"]) == units * 10
    assert len(scores["fp8_w"]) == len(scores["fp8_w8a8"]) == units * 10
    before = settings.answers_path.read_bytes()
    adapter_before = settings.adapter_path.read_bytes()
    assert worker.run_family(settings) == 0  # rerun: nothing regenerated, adapter kept
    assert settings.answers_path.read_bytes() == before
    assert settings.adapter_path.read_bytes() == adapter_before


def test_worker_regenerates_a_variant_when_the_temperature_changes(tmp_path: Path) -> None:
    settings = _tiny_settings(tmp_path)
    worker.run_family(settings)
    first = json.loads(settings.meta_path.read_text(encoding="utf-8"))["variants"]
    hotter = worker.RunSettings(**{**settings.__dict__, "default_temperature": 0.95,
                                   "do_scoring": False})
    worker.run_family(hotter)
    second = json.loads(hotter.meta_path.read_text(encoding="utf-8"))["variants"]
    assert second["plain"]["signature"]["temperature"] == 0.95
    assert second["lora07"]["signature"]["temperature"] == 0.95
    assert second["temp1"] == first["temp1"]  # fixed at 1.0: not regenerated
    assert second["plain"] != first["plain"]
    assert store.row_counts(hotter.answers_path)["zoo-qwen15-01"] == len(settings.bundle.units())


def test_worker_stops_cleanly_when_the_deadline_has_passed(tmp_path: Path) -> None:
    settings = _tiny_settings(tmp_path)
    expired = worker.RunSettings(**{**settings.__dict__, "deadline_epoch": 1.0,
                                    "do_scoring": False})
    assert worker.run_family(expired) == 3  # incomplete, nothing generated
    assert store.row_counts(expired.answers_path) == {}


def test_worker_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ZooConfigError):
        worker.parse_variants("plain,nonsense")
    assert worker.parse_variants("all") == GENERATION_ORDER
    save_tiny_family(tmp_path / "model")
    parser = worker.build_parser()
    bad_family = parser.parse_args(["run", "--family", "gemma", "--bundle", "b", "--out-dir", "o"])
    with pytest.raises(ZooConfigError):
        worker.settings_from_args(bad_family)
    no_model = parser.parse_args(["run", "--family", "qwen15", "--bundle", "b", "--out-dir", "o",
                                  "--model-dir", str(tmp_path / "nowhere")])
    with pytest.raises(worker.WorkerError):
        worker.settings_from_args(no_model)


def test_template_check_reports_a_tiny_family(tmp_path: Path) -> None:
    save_tiny_family(tmp_path / "qwen2.5-1.5b")
    qwen = next(f for f in FAMILIES if f.key == "qwen15")
    assert worker.check_templates(tmp_path, [qwen]) == 0
    assert worker.check_templates(tmp_path, [next(f for f in FAMILIES if f.key == "llama1")]) == 1
