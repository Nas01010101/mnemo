"""Stage 2 — LoRA SFT of a small Qwen2.5-instruct into a fact-distiller.

bf16 LoRA (no quantization — 0.5b/1.5b fit easily in 16GB and merge cleanly for GGUF).
Trains on the assistant completion only. Runs ON the RTX box in geofm_venv_cu124.

Usage: python train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --out ~/distiller_out/0.5b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

# The production distiller system prompt (kept in sync with src/tenet/distill.py _SYS).
SYS = """You extract durable, atomic facts from a message for an agent's long-term memory.
Return STRICT JSON: {"facts": [{"statement","key","salience","valid_at"}...]}.

Rules:
- statement: one self-contained fact. Resolve pronouns to names. No fluff.
  PRESERVE specific values VERBATIM — numbers, dates, times, durations, quantities,
  prices, proper nouns (e.g. keep "2 days", "March 3 at 14:20", "$50", "gate B12").
  Never generalize a specific away; those exact details are what gets asked about.
- key: a stable "subject::attribute" slug (lowercase, snake_case), e.g.
  "user::residence", "user::coffee_pref", "project_nimbus::ship_date". The SAME
  real-world attribute must always get the SAME key so later updates supersede it.
  CRITICAL: the account owner / first-person speaker ("I", "me", "my", and any name
  they give for themselves) is ALWAYS the subject `user` — never their proper name.
  So "I live in X", "I moved to Y", "My name is Z" all use subject `user`
  (keys user::residence, user::residence, user::name). This keeps updates on the
  same attribute colliding on one key so later values supersede earlier ones.
- salience: 0.0-1.0. Durable/identity/preference/commitment facts are high (0.7-1.0);
  transient small talk is low (0.0-0.3). Skip pure chit-chat entirely.
- valid_at: an ISO-8601 date/time if the fact states when it becomes true, else null.
- Extract nothing (empty list) if there is no durable fact worth remembering.
Return ONLY the JSON object."""


def load_rows(p: Path):
    rows = []
    for line in p.open():
        r = json.loads(line)
        rows.append({"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": r["text"]},
            {"role": "assistant", "content": json.dumps(r["target"], ensure_ascii=False)},
        ]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    data = Path(args.data)
    train_rows = load_rows(data / "train.jsonl")
    val_rows = load_rows(data / "val.jsonl")
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        bf16=True,
        max_length=1024,
        packing=False,
        assistant_only_loss=True,   # mask the prompt, train on the JSON completion
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=args.base,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    print("SAVED_ADAPTER", args.out, flush=True)


if __name__ == "__main__":
    main()
