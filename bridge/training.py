"""LoRA fine-tuning with the five training objectives.

All objectives are plain cross-entropy fine-tuning; they differ only in the
target of each example (no teacher model is involved).

    sft      every question on its T1 value
    vmd      as sft, but no loss on the (outdated) volatile answer
    avmd     volatile answers replaced by the deferral string d
    ca_vmd   volatile examples split in two halves: grounded (context
             prepended, target T2 value) and bare (no loss)
    ca_avmd  as ca_vmd, but the bare half is trained on d

Stable examples are trained on their answer in every objective.

The grounded / bare assignment uses Python's built-in hash(), which is salted
per process.  Set PYTHONHASHSEED (the drivers set it to the training seed) to
make the split reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from bridge.prompts import DEFERRAL_STRING

OBJECTIVES = ("sft", "vmd", "avmd", "ca_vmd", "ca_avmd")


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    objective: str
    deferral_string: str = DEFERRAL_STRING
    no_context_fraction: float = 0.5       # share of volatile examples in the bare half
    split_strategy: str = "per_example"    # "per_example" or "entity"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "o_proj", "down_proj")
    epochs: int = 3
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    seed: int = 42


class EvoWikiDataset(Dataset):
    """Tokenised (question, target) pairs with a per-token 'no loss' mask."""

    def __init__(self, data_path: str | Path, tokenizer, config: TrainingConfig):
        self.tokenizer = tokenizer
        self.cfg = config
        with open(data_path, encoding="utf-8") as f:
            self.examples = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.examples)

    def _use_context(self, ex: dict, idx: int) -> bool:
        key = ex.get("entity_key", ex["question"])
        h = hash(key) if self.cfg.split_strategy == "entity" else hash((key, idx))
        return h % 10000 >= int(self.cfg.no_context_fraction * 10000)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        obj = self.cfg.objective
        question, answer = ex["question"], ex["answer"]
        volatile = ex["fact_type"] == "evolved"
        user = question
        no_loss = False

        if volatile:
            if obj == "vmd":
                no_loss = True
            elif obj == "avmd":
                answer = self.cfg.deferral_string
            elif obj in ("ca_vmd", "ca_avmd"):
                if self._use_context(ex, idx):
                    user = f"Context: {ex.get('golden_context', '')}\n\n{question}"
                    answer = ex.get("latest_answer", answer)
                elif obj == "ca_avmd":
                    answer = self.cfg.deferral_string
                else:
                    no_loss = True

        full_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(full_text, truncation=True, max_length=self.cfg.max_seq_length,
                             return_tensors="pt", add_special_tokens=False)
        prompt_len = self.tokenizer(prompt_text, truncation=True,
                                    max_length=self.cfg.max_seq_length, return_tensors="pt",
                                    add_special_tokens=False)["input_ids"].shape[1]
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        if no_loss:
            labels[prompt_len:] = -100
        return {"input_ids": input_ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels}


class Collator:
    """Right-pads a batch; padding positions carry no loss."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        n = max(f["input_ids"].shape[0] for f in features)

        def pad(t, value):
            return torch.cat([t, torch.full((n - t.shape[0],), value, dtype=t.dtype)])

        return {
            "input_ids": torch.stack([pad(f["input_ids"], self.pad_id) for f in features]),
            "attention_mask": torch.stack([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.stack([pad(f["labels"], -100) for f in features]),
        }


def train(config: TrainingConfig, data_path: str | Path, output_dir: str | Path) -> Path:
    """Train one LoRA adapter and save it to <output_dir>/adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if config.objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {config.objective!r}; choose from {OBJECTIVES}")
    if config.objective in ("ca_vmd", "ca_avmd") and "PYTHONHASHSEED" not in os.environ:
        print("[train] WARNING: PYTHONHASHSEED is not set; the grounded/bare split "
              "will differ between runs.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # bf16/tf32 on Ampere or newer, fp16 otherwise (e.g. V100).
    ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if ampere else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_gpu = torch.cuda.device_count()
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=dtype, device_map="auto",
        max_memory={**{i: "14GiB" for i in range(n_gpu)}, "cpu": "80GiB"},
        trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=list(config.target_modules),
        bias="none"))
    model.print_trainable_parameters()

    dataset = EvoWikiDataset(data_path, tokenizer, config)
    print(f"[train] objective={config.objective} examples={len(dataset)}")
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=ampere, fp16=not ampere, tf32=ampere,
        logging_steps=10,
        save_strategy="epoch",
        seed=config.seed,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=config.objective in ("ca_vmd", "ca_avmd"),
    )
    Trainer(model=model, args=args, train_dataset=dataset,
            data_collator=Collator(tokenizer)).train()

    save_dir = output_dir / "adapter"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(config.__dict__, indent=2, default=list), encoding="utf-8")
    print(f"[train] adapter saved to {save_dir}")
    return save_dir
