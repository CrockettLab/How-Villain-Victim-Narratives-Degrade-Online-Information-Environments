"""
Applies the two fine-tuned classifiers to a dataset of headlines.
 
Loads the base model, attaches the fine-tuned LoRA adapters, and scores every
headline for villain presence and victim presence. Creates two probability columns.
 
To run:
python3 classify.py \
    --input  headlines.csv \
    --text_col headline \
    --adapters_villain ./olmo32b_villain_qlora/adapters \
    --adapters_victim  ./olmo32b_victim_qlora/adapters \
    --output headlines_classified.csv
 
The observational datasets were classified on a GPU cluster by running this script
over disjoint slices of the input; the scoring logic is identical.
"""
 
import argparse
import html as html_lib
import os
import re
import sys
 
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
 
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input CSV")
parser.add_argument("--text_col", default="headline")
parser.add_argument("--output", required=True)
parser.add_argument("--adapters_villain", required=True)
parser.add_argument("--adapters_victim", required=True)
parser.add_argument("--model_id", default="allenai/OLMo-2-0325-32B-Instruct")
parser.add_argument("--max_length", type=int, default=768)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--strip_html", action="store_true",
                    help="Remove HTML tags and unescape entities before scoring")
args = parser.parse_args()
 
 

# Prompt templates identical to the training data

def clean_headline(headline: str) -> str:
    return str(headline).replace("\n", " ").strip()
 
 
TAG_RE = re.compile(r"<[^>]+>")
 
 
def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(TAG_RE.sub(" ", str(text))))
 
 
def format_prompt_villain(headline: str) -> str:
    return f"""Below is a headline, and we would like you to tell us if it contains at least one identifiable villain. By 'villain,' we mean a person or group who intentionally causes harm to another person or group (their victim).
 
Consider the following example headline:
'Michigan Gov Gretchen Whitmer wants to extend COVID lockdown to punish protestors.'
 
This headline suggests that the governor is intentionally harming a group of protestors by punishing them with an extended COVID lockdown period. In this example, the governor is portrayed as a villain.
 
Note that we are not asking whether you believe that the governor is a villain. We are interested in whether the headline portrays a villain.
 
Villains do not have to be individuals. Groups of people, including organizations and countries, can also be portrayed as villains. A headline may contain multiple villains.
 
Here is the headline:
"{headline}"
 
Is there at least one identifiable villain in the headline? Respond either '1' if yes or '0' if no. Do not provide any other content in your response."""
 
 
def format_prompt_victim(headline: str) -> str:
    return f"""Below is a headline, and we would like you to tell us if it contains at least one identifiable victim. By 'victim,' we mean a person or group who is intentionally harmed by another person or group (a villain).
 
Consider the following example headline:
'Michigan Gov Gretchen Whitmer wants to extend COVID lockdown to punish protestors.'
 
This headline suggests that the governor is intentionally harming a group of protestors by punishing them with an extended COVID lockdown period. In this example, the protestors are portrayed as a victim.
 
Note that we are not asking whether you believe that the protestors are a victim. We are interested in whether the headline portrays a victim.
 
Victims do not have to be individuals. Groups of people, including organizations and countries, can also be portrayed as victims. A headline may contain multiple victims.
 
Here is the headline:
"{headline}"
 
Is there at least one identifiable victim in the headline? Respond either '1' if yes or '0' if no. Do not provide any other content in your response."""
 
 
PROMPT_FNS = {"villain": format_prompt_villain, "victim": format_prompt_victim}
 
 

# Scoring identical to the fine tuning script

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
 
 
@torch.inference_mode()
def score_texts(model, tok, texts, prompt_fn, ids0, ids1):
    """Return P(class = 1) for each text, in the order given."""
    probs = np.full(len(texts), np.nan)
    # Sort by length so that each batch pads to a similar width.
    order = np.argsort([len(t) for t in texts], kind="stable")
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        enc = tok([prompt_fn(texts[i]) for i in idx], return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=args.max_length).to(model.device)
        logits = model(**enc).logits
        last_pos = (enc["attention_mask"].sum(dim=1) - 1).clamp(min=0)
        last = logits[torch.arange(logits.size(0), device=logits.device), last_pos, :]
        lse0 = torch.logsumexp(last[:, ids0], dim=-1)
        lse1 = torch.logsumexp(last[:, ids1], dim=-1)
        probs[idx] = torch.softmax(torch.stack([lse0, lse1], dim=-1),
                                   dim=-1)[:, 1].float().cpu().numpy()
    return probs
 
 

# Load and prepare text
df = pd.read_csv(args.input, low_memory=False)
if args.text_col not in df.columns:
    sys.exit(f"Column '{args.text_col}' not found. Columns: {df.columns.tolist()}")
 
n_rows = len(df)
text = df[args.text_col].fillna("")          
if args.strip_html:                         
    text = text.map(strip_html)
text_clean = text.map(clean_headline)
 
valid = text_clean.str.len() > 0
uniq_texts = pd.Index(text_clean[valid].unique()).tolist()
pos = {t: i for i, t in enumerate(uniq_texts)}
row_to_uniq = np.array([pos[t] if v else -1
                        for t, v in zip(text_clean, valid)], dtype=np.int64)
print(f"{n_rows} rows | {len(uniq_texts)} unique headlines | "
      f"{int((~valid).sum())} empty (scored as NaN)")
 
 

# Load the base model once, attach both fine-tunedadapters, score each task in turn

tok = AutoTokenizer.from_pretrained(args.model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
 
base = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto",
)
model = PeftModel.from_pretrained(base, args.adapters_villain, adapter_name="villain")
model.load_adapter(args.adapters_victim, adapter_name="victim")
model.eval()
model.config.use_cache = False
 
ids0, ids1 = digit_token_sets(tok)
 
for task in ("villain", "victim"):
    print(f"Scoring: {task}", flush=True)
    model.set_adapter(task)
    uniq_probs = score_texts(model, tok, uniq_texts, PROMPT_FNS[task], ids0, ids1)
    df[f"{task}_prob"] = np.where(row_to_uniq >= 0, uniq_probs[row_to_uniq], np.nan)
 
assert len(df) == n_rows
df.to_csv(args.output, index=False)
print(f"Saved: {os.path.abspath(args.output)}")