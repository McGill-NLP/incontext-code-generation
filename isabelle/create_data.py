import json
import pandas as pd

df = pd.read_json('aligned_problems/new_complete_mini.jsonl', lines=True)
df['informal_statement'] = df['informal_statement'].apply(json.dumps)
df['informal_proof'] = df['informal_proof'].apply(json.dumps)
df['formal_statement'] = df['formal_statement'].apply(json.dumps)

df.to_csv("data.tsv", sep = '\t', index = None)