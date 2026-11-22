"""Model Equality zoo (team F): a public-model replica of the 9 `ref-*` endpoints, per family.

The 9 reference endpoints are ONE small model (SmolLM2-135M) served in nine ways: plain, fp8,
system prompt, persona, watermark, temperature 1.0 and three LoRA strengths. The `ep-*` endpoints
are other families (Qwen, Llama...). The zoo rebuilds the same nine variants on open models that
fit one A100 of the team node, so that the frozen decision rules of teams C and D can be measured
on families they were NOT tuned on, before a single scored `ep-*` answer is spent.

Layout:
    config      names, seeds, variants and the family registry (no number lives elsewhere)
    prompts     the prompt bundle shipped to the cluster (arithmetic, identity, system prompts)
    quant       E4M3 fake quantisation (weights per tensor, activations per tensor)
    lora        `ZooLinear` (LoRA with a runtime scale + fp8) and the short LoRA training
    sampling    chat prompts, chunked sampling with out-of-memory fallback, KGW watermark
    scoring     per-token log-likelihood of an answer under a model (chat template, no system)
    store       atomic per-variant upsert of the official-format answers file
    worker      one family on one GPU: generation, LoRA, fp8, likelihood scoring (cluster side)
    calibrate   estimate of the server default temperature from the `ref` answers (cluster side)
    evaluate    frozen rules v2 / v3 / likelihood flag on the zoo answers (Mac side)

Runtime constraint (cluster side): only `torch`, `transformers`, `numpy`, `safetensors`, the
standard library and, for the KGW module, `scipy`. No pydantic: frozen dataclasses with explicit
validation, as in the rest of `hackathon`.
"""
