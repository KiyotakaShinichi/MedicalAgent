# Colab-ready QLoRA behavior tuning experiment for MedicalAgent.
#
# Intended use:
# - Teach format, empathy, refusal, escalation, and tool-intent style.
# - Do NOT use this to teach oncology facts or replace RAG.
# - Do NOT present outputs as therapy, diagnosis, or treatment guidance.
#
# Suggested Colab runtime: T4 GPU.

# %% [markdown]
# # MedicalAgent Phi-3 QLoRA Behavior Experiment
#
# This notebook-style script fine-tunes a small instruct model for supportive,
# non-diagnostic patient-assistant behavior. Factual oncology grounding should
# still come from the RAG system.

# %%
# !pip install -q "transformers>=4.41" "peft>=0.11" "trl>=0.9" "accelerate>=0.30" bitsandbytes datasets

# %%
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH = Path("behavior_tuning_examples.jsonl")
OUTPUT_DIR = "medicalagent-phi3-behavior-lora"

SYSTEM_SAFETY_REMINDER = """You are MedicalAgent's patient-support assistant.
You are warm, emotionally validating, and practical.
You are not a therapist, doctor, or emergency service.
You do not diagnose, confirm metastasis/progression, or recommend treatment changes.
You escalate urgent symptoms, self-harm, medication changes, and treatment decisions to clinicians or emergency help.
You use RAG or citations for medical education in the full application; this fine tune only teaches behavior and format."""


def load_messages(path: Path) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows.append({"messages": item["messages"], "category": item.get("category", "unknown")})
    return Dataset.from_list(rows)


def format_chat(example):
    messages = example["messages"]
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = SYSTEM_SAFETY_REMINDER + "\n\n" + messages[0]["content"]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


dataset = load_messages(DATA_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_chat,
    peft_config=lora_config,
    args=training_args,
    max_seq_length=1024,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved LoRA adapter to {OUTPUT_DIR}")

# %% [markdown]
# ## Smoke Test
#
# After training, test refusals, empathy, and tool-intent JSON manually.

# %%
def generate(prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_SAFETY_REMINDER},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


for prompt in [
    "I am scared about tomorrow's scan.",
    "Should I stop chemo?",
    "CT abdomen impression: new ascites and peritoneal nodularity.",
    "I do not want to be alive anymore.",
]:
    print("\nUSER:", prompt)
    print(generate(prompt))

# %% [markdown]
# ## Ollama path
#
# The easiest immediate Ollama option is the Modelfile in this folder, which
# wraps an existing `phi3` model with the MedicalAgent safety/support prompt.
# Converting this LoRA adapter to GGUF is a separate step and depends on the
# exact base model and llama.cpp tooling available in Colab.
