"""Model loading and greedy generation.

Every run in the paper uses fp16 weights, the LoRA adapter merged into the
base model, the backbone's chat template, and greedy decoding.

Placement note: CPU-offloaded fp16 layers change the numerics enough to flip
a small fraction of near-tie greedy decodes (about 0.5% of outputs).  Keep the
whole model on GPU when comparing runs; the per-GPU budget is read from
BRIDGE_MAX_GPU_MEM (default 14GiB, which fits an 8B model across two 16GB
cards).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, adapter_path: str | None = None,
               torch_dtype=torch.float16, max_memory_per_gpu: str | None = None):
    """Return (tokenizer, model) with an optional LoRA adapter merged in."""
    from accelerate import dispatch_model, infer_auto_device_map

    max_memory_per_gpu = max_memory_per_gpu or os.environ.get("BRIDGE_MAX_GPU_MEM", "14GiB")
    if adapter_path is None and Path(model_id).is_dir() and \
            (Path(model_id) / "adapter_config.json").exists():
        from peft import PeftConfig
        adapter_path = model_id
        model_id = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load on CPU first so merge_and_unload is not blocked by dispatch hooks.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

    cpu_budget = os.environ.get("BRIDGE_MAX_CPU_MEM") or f"{_available_cpu_gib()}GiB"
    memory = {i: max_memory_per_gpu for i in range(torch.cuda.device_count())}
    memory["cpu"] = cpu_budget
    device_map = infer_auto_device_map(
        model, max_memory=memory,
        no_split_module_classes=list(getattr(model, "_no_split_modules", None) or []))
    model = dispatch_model(model, device_map=device_map)
    model.eval()

    placements = sorted({str(d) for d in device_map.values()})
    print(f"[model] Model loaded: {model_id} adapter={adapter_path} placements={placements}")
    if any(d in ("cpu", "disk", "meta") for d in placements):
        print("[model] WARNING: some layers are not on GPU; greedy decodes may "
              "differ slightly from fully GPU-resident runs.")
    return tokenizer, model


def _available_cpu_gib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return max(4, int(int(line.split()[1]) / 1024 ** 2 * 0.6))
    except OSError:
        pass
    return 8


def get_layers(model: torch.nn.Module):
    """The list of transformer blocks (Llama, Mistral and Qwen layouts)."""
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("Cannot locate the transformer layer list.")


def resolve_layer(model: torch.nn.Module, layer: int) -> int:
    return layer if layer >= 0 else len(get_layers(model)) + layer


def input_device(model) -> torch.device:
    try:
        return model.model.embed_tokens.weight.device
    except AttributeError:
        return next(model.parameters()).device


def chat_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    """Token ids of one user turn plus the generation prompt, shape (1, T)."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt", add_generation_prompt=True)
    ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
    return ids.unsqueeze(0) if ids.dim() == 1 else ids


@torch.inference_mode()
def generate(prompt: str, tokenizer, model, max_new_tokens: int = 64) -> str:
    """Greedy completion of one prompt wrapped in the chat template."""
    input_ids = chat_input_ids(tokenizer, prompt).to(input_device(model))
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    del input_ids, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text
