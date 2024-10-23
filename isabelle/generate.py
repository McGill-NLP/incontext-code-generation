import os
import json
import argparse
import datetime
import random
import torch
import pandas as pd
import pdb
from IPython.core.display import HTML
from functools import partial
import tiktoken

import openai
import sys

from models import LargeLanguageModel
from prompt_creator import get_prompt

def build_parser():
	parser = argparse.ArgumentParser(description='Evaluate LLMs on Isabelle')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-data', type=str, default='data.tsv', help='data')
	parser.add_argument('-out_dir', type=str, default='results/', help='Output Directory')
	parser.add_argument('-stop', type=list, default=['Informal'], help='When to stop generation')
	parser.add_argument('-prompt_type', type=str, default='dsp', help='prompt type')
	parser.add_argument('-num_ex', type=int, default=8, help='Number of examples to give for demos')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-4o-mini', help='Which model to use')
	parser.add_argument('-port', type=int, default=8080, help='Port on which the model is hosted')
	parser.add_argument('-timeout', type=int, default=1000000, help='Timeout for the model')
	parser.add_argument('-batch_size', type=int, default=1, help='Batch Size')
	parser.add_argument('-max_tokens', type=int, default=256, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.5, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	return parser

keyword_alias = {
	"theorem": "abcx",
	"fixes": "defy",
	"assumes": "ghiz",
	"and": "jklx",
	"shows": "mnoy",
	"have": "pqrz",
	"using": "stux",
	"sledgehammer": "vwxy",
	"then": "yzaz",
	"show": "cefx",
	"?thesis": "hijy",
	"qed": "klmz"
}
alias_keyword = {
	"abcx": "theorem",
	"defy": "fixes",
	"ghiz": "assumes",
	"jklx": "and",
	"mnoy": "shows",
	"pqrz": "have",
	"stux": "using",
	"vwxy": "sledgehammer",
	"yzaz": "then",
	"cefx": "show",
	"hijy": "?thesis",
	"klmz": "qed"
}

def map_aliases(pred):
	for keyword in alias_keyword:
		pred = pred.replace(keyword, alias_keyword[keyword])
	return pred

def post_process(pred, prob):
	pred = pred.replace("```", "")
	pred = pred.strip()
	pred = pred.replace("Informal", "")
	entire = prob + "\nproof -\n  " + pred.strip()
	return entire.strip()

def main(args):
	print("Beginning Generation...")

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty, port=args.port, timeout=args.timeout)

	df = pd.read_csv(args.data, sep = '\t')

	pred_ls = []

	for i in range(len(df)):
		ex = df.loc[i]
		name = ex['problem_name']
		prob = json.loads(ex['formal_statement'])

		if "mathd_algebra" not in name:
			continue

		prompt, sys_prompt = get_prompt(ex, args.prompt_type, args.num_ex, keyword_alias, alias_keyword)

		if args.model == "gpt-3.5-turbo":
			encoding = tiktoken.encoding_for_model(args.model)
			pr_len = len(encoding.encode(prompt))
			while pr_len + args.max_tokens > 4096:
				prompt, sys_prompt = get_prompt(ex, args.prompt_type, args.num_ex, keyword_alias, alias_keyword)
				pr_len = len(encoding.encode(prompt))

		if prompt == "PROMPT_EXAMPLE":
			continue

		og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
		
		while og_pred == "ERROR":
			prompt, sys_prompt = get_prompt(ex, args.prompt_type, args.num_ex, keyword_alias, alias_keyword)
			og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)

		if "alias" in args.prompt_type:
			og_pred = map_aliases(og_pred)

		pred = post_process(og_pred, prob)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Name: " + name + "\n")
			f.write("Problem:\n" + prob + "\n\n")
			f.write("Prediction:\n" + pred + "\n")
			f.write("========================================================================\n")

		pred_ls.append([name, prob, pred])

		pred_df = pd.DataFrame(pred_ls, columns = ['Name', 'Problem', 'Prediction'])
		pred_df['Problem'] = pred_df['Problem'].apply(json.dumps)
		pred_df['Prediction'] = pred_df['Prediction'].apply(json.dumps)
		pred_df.to_csv(args.out_dir + "/predictions.tsv", sep = '\t', index = None)

		print("Completed {} / {}...".format(i+1, len(df)), end = '\r', flush = True)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.prompt_type + "-num-ex-" + str(args.num_ex) + "_" + args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_KEY")

	main(args)