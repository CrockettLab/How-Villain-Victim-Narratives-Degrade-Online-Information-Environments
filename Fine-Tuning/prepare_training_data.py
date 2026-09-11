"""
Curates the fine-tuning villain and victim datasets.
 
Takes the headline-level annotation dataset, restricts it to headlines a
majority of annotators rated as clear, splits it into training and held-out
test sets, and writes prompt/response JSONL files for each task.
 
Outputs:
train_formatted_villain.jsonl, test_formatted_villain.jsonl
train_formatted_victim.jsonl,  test_formatted_victim.jsonl
train_test_split_record.csv    (headline ids + split, used later for calibration)
"""
 
import json
import pandas as pd
from sklearn.model_selection import train_test_split
 
DATA_PATH = "headline_level_data.csv"
TEST_SIZE = 0.20
SEED = 42
 

def clean_headline(headline: str) -> str:
    return str(headline).replace("\n", " ").strip()
 
 
def format_prompt_villain(headline: str) -> str:
    return f"""Below is a headline, and we would like you to tell us if it contains at least one identifiable villain. By 'villain,' we mean a person or group who intentionally causes harm to another person or group (their victim).
 
Consider the following example headline:
'Michigan Gov Gretchen Whitmer wants to extend COVID lockdown to punish protestors.'
 
This headline suggests that the governor is intentionally harming a group of protestors by punishing them with an extended COVID lockdown period. In this example, the governor is portrayed as a villain.
 
Note that we are not asking whether you believe that the governor is a villain. We are interested in whether the headline portrays a villain.
 
Villains do not have to be individuals. Groups of people, including organizations and countries, can also be portrayed as villains. A headline may contain multiple villains.
 
Here is the headline:
"{clean_headline(headline)}"
 
Is there at least one identifiable villain in the headline? Respond either '1' if yes or '0' if no. Do not provide any other content in your response."""
 
 
def format_prompt_victim(headline: str) -> str:
    return f"""Below is a headline, and we would like you to tell us if it contains at least one identifiable victim. By 'victim,' we mean a person or group who is intentionally harmed by another person or group (a villain).
 
Consider the following example headline:
'Michigan Gov Gretchen Whitmer wants to extend COVID lockdown to punish protestors.'
 
This headline suggests that the governor is intentionally harming a group of protestors by punishing them with an extended COVID lockdown period. In this example, the protestors are portrayed as a victim.
 
Note that we are not asking whether you believe that the protestors are a victim. We are interested in whether the headline portrays a victim.
 
Victims do not have to be individuals. Groups of people, including organizations and countries, can also be portrayed as victims. A headline may contain multiple victims.
 
Here is the headline:
"{clean_headline(headline)}"
 
Is there at least one identifiable victim in the headline? Respond either '1' if yes or '0' if no. Do not provide any other content in your response."""
 
 
TASKS = {
    "villain": (format_prompt_villain, "villain_presence_majority"),
    "victim":  (format_prompt_victim,  "victim_presence_majority"),
}
 
 
def write_jsonl(frame: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        for _, row in frame.iterrows():
            f.write(json.dumps({"prompt": row["prompt"],
                                "response": row["response"]}) + "\n")
 
 

# Load headline-level annotation data and subset on headlines a majority of annotators rated as clear
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df[df["clarity_majority"] == 1].copy().reset_index(drop=True)
print(f"Clear headlines: {len(df)}")
 
 

# Generate a stratified train/test split
# Stratification is on the JOINT villain x victim label so that both
# classifiers see the same split with the same class balance in each partition.
joint = (df["villain_presence_majority"].astype(str)
         + df["victim_presence_majority"].astype(str))
 
train_idx, test_idx = train_test_split(
    df.index, test_size=TEST_SIZE, stratify=joint, random_state=SEED
)
 
df["split"] = "train"
df.loc[test_idx, "split"] = "test"
 
df[["url_rid", "share_title", "split",
    "villain_presence_majority", "victim_presence_majority"]].to_csv(
    "train_test_split_record.csv", index=False
)
 
 

# Format prompts and write one JSONL per task per split

for task, (fmt, label_col) in TASKS.items():
    out = df.copy()
    out["prompt"] = out["share_title"].map(fmt)
    out["response"] = out[label_col].astype(str)
 
    for split in ("train", "test"):
        sub = out[out["split"] == split]
        path = f"{split}_formatted_{task}.jsonl"
        write_jsonl(sub, path)
        print(f"{task:8s} {split:5s} n={len(sub):5d}  "
              f"positive_rate={(sub['response'] == '1').mean():.3f}  -> {path}")