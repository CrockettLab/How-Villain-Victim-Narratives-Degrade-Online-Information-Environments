"""
Performs QLoRA fine-tuning of OLMo-2-32B-Instruct for binary villain/victim headline
classification, and runs an evaluation on the held-out test set.
 
To run:
python3 fine_tune.py --task villain
python3 fine_tune.py --task victim
 
Requires:
train_formatted_<task>.jsonl and test_formatted_<task>.jsonl from
prepare_training_data.py in --data_dir.
"""
 
import argparse
import json
import os
import random
 
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model
 
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["villain", "victim"], required=True)
parser.add_argument("--data_dir", default=".")
args = parser.parse_args()
 
MODEL_NAME   = "allenai/OLMo-2-0325-32B-Instruct"
TRAIN_JSONL  = os.path.join(args.data_dir, f"train_formatted_{args.task}.jsonl")
TEST_JSONL   = os.path.join(args.data_dir, f"test_formatted_{args.task}.jsonl")
OUT_DIR      = f"./olmo32b_{args.task}_qlora"
VAL_FRACTION = 0.15
MAX_LENGTH   = 768
THRESHOLD    = 0.5
SEED         = 42
 
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
 
 

# Tokenization
# Only supervise the response token, mask the prompt with -100

def map_label(ex):
    return {"label": 1 if str(ex["response"]).strip() == "1" else 0}
 
 
def make_preprocess(tok, max_length):
    def preprocess(ex):
        p_ids = tok(ex["prompt"],   add_special_tokens=False)["input_ids"]
        r_ids = tok(ex["response"], add_special_tokens=False)["input_ids"]
        p_ids = p_ids[:max(1, max_length - max(1, len(r_ids)))]
        full = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        return {"input_ids": full,
                "attention_mask": [1] * len(full),
                "labels": labels,
                "label": ex.get("label")}
    return preprocess
 
 
def make_collator(tok):
    def collate(features):
        batch = tok.pad(
            [{k: f[k] for k in ("input_ids", "attention_mask")} for f in features],
            padding=True, return_tensors="pt", pad_to_multiple_of=8,
        )
        max_len = batch["input_ids"].size(1)
        batch["labels"] = torch.tensor(
            [f["labels"] + [-100] * (max_len - len(f["labels"])) for f in features],
            dtype=torch.long,
        )
        return batch
    return collate
 
 

# Scoring
# P(class = 1) from the logits at the final position

def digit_token_sets(tokenizer):
    def collect(digit):
        ids = set()
        for s in (digit, " " + digit, "\n" + digit, "\t" + digit):
            enc = tokenizer.encode(s, add_special_tokens=False)
            if enc and digit in tokenizer.convert_ids_to_tokens([enc[-1]])[0]:
                ids.add(enc[-1])
        return ids
    ids0, ids1 = collect("0"), collect("1")
    ids0, ids1 = ids0 - ids1, ids1 - ids0
    assert not (ids0 & ids1), "class token sets must be disjoint"
    return sorted(ids0), sorted(ids1)
 
 
@torch.no_grad()
def batched_probs(model, tok, prompts, ids0, ids1, batch_size=16):
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LENGTH).to(model.device)
        logits = model(**enc).logits
        idx = (enc["attention_mask"].sum(dim=1) - 1).clamp(min=0)
        last = logits[torch.arange(logits.size(0), device=logits.device), idx, :]
        lse0 = torch.logsumexp(last[:, ids0], dim=-1)
        lse1 = torch.logsumexp(last[:, ids1], dim=-1)
        p = torch.softmax(torch.stack([lse0, lse1], dim=-1), dim=-1)[:, 1]
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)
 
 

# Data
# raining partition is split again into train / validation.
# The validation set is used only for early stopping and checkpoint selection;
# the test set is never seen during training or model selection.

train_df = load_dataset("json", data_files=TRAIN_JSONL, split="train").map(map_label).to_pandas()
test_df  = load_dataset("json", data_files=TEST_JSONL,  split="train").map(map_label).to_pandas()
 
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
idx_trn, idx_val = next(sss.split(train_df, train_df["label"]))
trn_df = train_df.iloc[idx_trn].reset_index(drop=True)
val_df = train_df.iloc[idx_val].reset_index(drop=True)
print(f"train={len(trn_df)}  val={len(val_df)}  test={len(test_df)}")
 
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
 
preprocess = make_preprocess(tok, MAX_LENGTH)
train_sft = Dataset.from_pandas(trn_df).map(preprocess, remove_columns=list(trn_df.columns))
val_sft   = Dataset.from_pandas(val_df).map(preprocess, remove_columns=list(val_df.columns))
 

# 4-bit base model with LoRA adapters on the attention projections (QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto",
)
model.gradient_checkpointing_enable()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
))
model.enable_input_require_grads()
model.config.use_cache = False
model.print_trainable_parameters()
 
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,      
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED, bf16=True, gradient_checkpointing=True, report_to="none",
    ),
    train_dataset=train_sft,
    eval_dataset=val_sft,
    processing_class=tok,
    data_collator=make_collator(tok),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
 
model.save_pretrained(os.path.join(OUT_DIR, "adapters"))
tok.save_pretrained(OUT_DIR)
 
 
# Held-out evaluation
ids0, ids1 = digit_token_sets(tok)
test_y = test_df["label"].to_numpy()
test_p = batched_probs(model, tok, test_df["prompt"].tolist(), ids0, ids1)
test_pred = (test_p >= THRESHOLD).astype(int)
 
metrics = {
    "task": args.task,
    "threshold": THRESHOLD,
    "n_test": int(len(test_y)),
    "accuracy": float(accuracy_score(test_y, test_pred)),
    "mcc": float(matthews_corrcoef(test_y, test_pred)),
    "auroc": float(roc_auc_score(test_y, test_p)),
    "brier": float(brier_score_loss(test_y, test_p)),
    "majority_baseline_accuracy": float(accuracy_score(test_y, np.zeros_like(test_y))),
    "confusion_matrix": confusion_matrix(test_y, test_pred).tolist(),
    "classification_report": classification_report(test_y, test_pred, digits=3,
                                                   zero_division=0),
}
print(json.dumps({k: v for k, v in metrics.items()
                  if k != "classification_report"}, indent=2))
print(metrics["classification_report"])
 
json.dump(metrics, open(os.path.join(OUT_DIR, "test_metrics.json"), "w"), indent=2)
 
# Save probabilities for the source-quality calibration step.
val_p = batched_probs(model, tok, val_df["prompt"].tolist(), ids0, ids1)
pd.DataFrame({"prompt": val_df["prompt"], "label": val_df["label"],
              "prob_1": val_p}).to_csv(os.path.join(OUT_DIR, "val_probs.csv"), index=False)
pd.DataFrame({"prompt": test_df["prompt"], "label": test_y,
              "prob_1": test_p}).to_csv(os.path.join(OUT_DIR, "test_probs.csv"), index=False)
print(f"Saved to {os.path.abspath(OUT_DIR)}")