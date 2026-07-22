"""Explicitly gated QLoRA experiment for NLCare behavior formatting.

Default execution performs preflight only. Real training requires ``--execute``
plus pinned base-model and tokenizer revisions. It consumes only the governed
training split and never selects a checkpoint on the internal frozen holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "Data" / "finetune" / "prepared" / "dataset_train.jsonl"
DEFAULT_DEV = ROOT / "Data" / "finetune" / "prepared" / "dataset_development.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "qlora_behavior" / "adapter_output"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_split(path: Path, expected_split: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest_path = path.parent / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("split_manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details = (manifest.get("splits") or {}).get(expected_split)
    if not details or Path(str(details.get("path"))).name != path.name:
        raise ValueError(f"{path.name} is not the declared {expected_split} split")
    actual = sha256_file(path)
    if details.get("sha256") != actual:
        raise ValueError(f"{expected_split} split hash mismatch")
    return {"path": str(path), "sha256": actual, "example_count": details["example_count"]}


def preflight(args: argparse.Namespace) -> dict:
    if args.train.resolve() == args.development.resolve():
        raise ValueError("Training and development paths must be different.")
    train = verify_split(args.train, "train")
    development = verify_split(args.development, "development")
    checks = {
        "base_model_revision_pinned": bool(args.base_revision),
        "tokenizer_revision_pinned": bool(args.tokenizer_revision),
        "license_review_acknowledged": bool(args.license_reviewed),
        "train_hash_verified": True,
        "development_hash_verified": True,
        "frozen_holdout_used_for_training_or_selection": False,
    }
    execution_controls_ready = (
        checks["base_model_revision_pinned"]
        and checks["tokenizer_revision_pinned"]
        and checks["license_review_acknowledged"]
        and checks["train_hash_verified"]
        and checks["development_hash_verified"]
        and not checks["frozen_holdout_used_for_training_or_selection"]
    )
    return {
        "schema_version": "qlora_execution_preflight_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ready_for_explicit_experimental_execution"
            if execution_controls_ready
            else "blocked"
        ),
        "clinical_validation": False,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "train": train,
        "development": development,
        "checks": checks,
        "model_trained": False,
        "promotion_decision": "HOLD",
        "claim_boundary": (
            "Behavior-only synthetic experiment. Execution does not authorize "
            "patient-facing use or create medical knowledge or clinical evidence."
        ),
    }


def execute(args: argparse.Namespace, preflight_report: dict) -> None:
    if preflight_report["status"] != "ready_for_explicit_experimental_execution":
        raise RuntimeError("Preflight is blocked; pin revisions and acknowledge license review.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        set_seed,
    )
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Experimental QLoRA execution requires a CUDA GPU.")
    set_seed(20260721)

    def load_dataset(path: Path) -> Dataset:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                rows.append({"messages": payload["messages"], "id": payload["id"]})
        return Dataset.from_list(rows)

    train_dataset = load_dataset(args.train)
    development_dataset = load_dataset(args.development)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.tokenizer_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    def format_chat(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=0.0001,
        warmup_ratio=0.05,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        seed=20260721,
        data_seed=20260721,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=development_dataset,
        formatting_func=format_chat,
        peft_config=lora,
        args=training_args,
        max_seq_length=1536,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    execution = {
        **preflight_report,
        "status": "experimental_adapter_trained_not_evaluated",
        "model_trained": True,
        "promotion_decision": "HOLD",
        "next_step": "Generate complete baseline and candidate outputs, then run the promotion gate.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_execution_manifest.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLCare governed QLoRA experiment.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--base-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--license-reviewed", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = preflight(arguments)
    print(json.dumps(report, indent=2))
    if arguments.execute:
        execute(arguments, report)
