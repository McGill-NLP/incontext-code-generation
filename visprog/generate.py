import os
import argparse
import datetime
import random
import torch
import pandas as pd
import pdb
from IPython.core.display import HTML
from functools import partial
import tiktoken

import sys
module_path = os.path.abspath(os.path.join('visprog_code'))
if module_path not in sys.path:
    sys.path.append(module_path)

from engine.utils import ProgramGenerator
from prompts.nlvr import create_prompt as create_nlvr_prompt
from prompts.gqa import create_prompt as create_gqa_prompt
from prompts.knowtag import create_prompt as create_knowtag_prompt
from prompts.imgedit import create_prompt as create_imgedit_prompt

def build_parser():
	parser = argparse.ArgumentParser(description='Evaluate LLMs on Vision')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='results/', help='Output Directory')
	parser.add_argument('-stop', type=list, default=['Question', 'Statement', 'Instruction', "```"], help='When to stop generation')
	parser.add_argument('-data', type=str, default='gqa', help='data')
	parser.add_argument('-prompt_type', type=str, default='demos', help='prompt type')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-4o-mini', help='Which model to use')
	parser.add_argument('-port', type=int, default=8080, help='Port on which the model is hosted')
	parser.add_argument('-timeout', type=int, default=1000000, help='Timeout for the model')
	parser.add_argument('-batch_size', type=int, default=1, help='Batch Size')
	parser.add_argument('-max_tokens', type=int, default=256, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.7, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	return parser

def generate_gqa(args):
	prompter = partial(create_gqa_prompt, prompt_type=args.prompt_type, method='all')
	test_df = pd.read_csv('visprog_code/data/gqa/test.tsv', sep='\t')

	generator = ProgramGenerator(prompter=prompter, model_type=args.model_type, model_name=args.model, temperature=args.temperature, top_p=args.top_p, stop=args.stop, n=args.n, max_tokens=args.max_tokens, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty, port=args.port, timeout=args.timeout)

	if args.model_type in ['chat', 'completion']:
		encoding = tiktoken.encoding_for_model(args.model)
	else:
		encoding = None
	tot_input_tokens = 0
	tot_output_tokens = 0

	programs_ls = []

	print("Beginning Test for GQA...")

	tot_input_tokens = 0
	tot_output_tokens = 0

	programs_ls = []

	for i in range(len(test_df)):
		index = test_df.loc[i]['index']
		question = test_df.loc[i]['question']
		answer = test_df.loc[i]['answer']

		with open(args.out_dir + "/test_logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(index) + "\n")
			f.write("Question: " + question + "\n")
			f.write("Ground Truth Answer: " + str(answer) + "\n")

		_, prog, (cur_input_tokens, cur_output_tokens) = generator.generate(dict(question=question), tik_encoding=encoding)

		tot_input_tokens += cur_input_tokens
		tot_output_tokens += cur_output_tokens

		programs_ls.append([index, question, answer, prog])
		programs_df = pd.DataFrame(programs_ls, columns=['Index', 'Question', 'Answer', 'Program'])
		programs_df.to_csv(args.out_dir + "/test_generated_programs.csv", index=False)

		with open(args.out_dir + "/test_logs.txt", "a") as f:
			f.write("Generated Program:\n" + prog  + "\n")

		print("Completed {}/{}".format(i, len(test_df)), end = '\r', flush = True)

		print("Total Input Tokens: " + str(tot_input_tokens))
		print("Total Output Tokens: " + str(tot_output_tokens))

def generate_nlvr(args):
	prompter = partial(create_nlvr_prompt, prompt_type=args.prompt_type, method='all')
	test_df = pd.read_json('visprog_code/data/nlvr/nlvr2/data/test1.json', lines=True)

	generator = ProgramGenerator(prompter=prompter, model_type=args.model_type, model_name=args.model, temperature=args.temperature, top_p=args.top_p, stop=args.stop, n=args.n, max_tokens=args.max_tokens, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty, port=args.port, timeout=args.timeout)

	if args.model_type in ['chat', 'completion']:
		encoding = tiktoken.encoding_for_model(args.model)
	else:
		encoding = None

	tot_input_tokens = 0
	tot_output_tokens = 0

	programs_ls = []

	print("Beginning Test for NLVR...")

	for i in range(len(test_df)):
		sentence = test_df.loc[i]['sentence']
		label = test_df.loc[i]['label']

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(i) + "\n")
			f.write("Sentence: " + sentence + "\n")
			f.write("Ground Truth Label: " + str(label) + "\n")

		_, prog, (cur_input_tokens, cur_output_tokens) = generator.generate(dict(statement=sentence), tik_encoding=encoding)
		
		tot_input_tokens += cur_input_tokens
		tot_output_tokens += cur_output_tokens

		programs_ls.append([i, sentence, label, prog])
		programs_df = pd.DataFrame(programs_ls, columns=['Example', 'Sentence', 'Label', 'Program'])
		programs_df.to_csv(args.out_dir + "/generated_programs.csv", index=False)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Generated Program:\n" + prog  + "\n")

		print("Completed {}/{}".format(i, len(test_df)), end = '\r', flush = True)

	print("Total Input Tokens: " + str(tot_input_tokens))
	print("Total Output Tokens: " + str(tot_output_tokens))

def generate_knowtag(args):
	prompter = partial(create_knowtag_prompt, prompt_type=args.prompt_type, method='all')
	
	test_df = pd.read_csv('visprog_code/data/knowtag/modified_instructions.csv')

	generator = ProgramGenerator(prompter=prompter, model_type=args.model_type, model_name=args.model, temperature=args.temperature, top_p=args.top_p, stop=args.stop, n=args.n, max_tokens=args.max_tokens, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty, port=args.port, timeout=args.timeout)

	if args.model_type in ['chat', 'completion']:
		encoding = tiktoken.encoding_for_model(args.model)
	else:
		encoding = None

	tot_input_tokens = 0
	tot_output_tokens = 0

	programs_ls = []

	print("Beginning Test for KnowTag...")

	for i in range(len(test_df)):
		instruction = test_df.loc[i]['instruction']
		image_name = test_df.loc[i]['image_name']

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(i) + "\n")
			f.write("Instruction: " + instruction + "\n")

		_, prog, (cur_input_tokens, cur_output_tokens) = generator.generate(dict(instruction=instruction, list_max=20), tik_encoding=encoding)
		
		tot_input_tokens += cur_input_tokens
		tot_output_tokens += cur_output_tokens

		programs_ls.append([i, instruction, image_name, prog])
		programs_df = pd.DataFrame(programs_ls, columns=['Example', 'Instruction', 'Image', 'Program'])
		programs_df.to_csv(args.out_dir + "/generated_programs.csv", index=False)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Generated Program:\n" + prog  + "\n")

		print("Completed {}/{}".format(i, len(test_df)), end = '\r', flush = True)

	print("Total Input Tokens: " + str(tot_input_tokens))
	print("Total Output Tokens: " + str(tot_output_tokens))

def generate_imgedit(args):
	prompter = partial(create_imgedit_prompt, prompt_type=args.prompt_type, method='all')
	
	test_df = pd.read_csv('visprog_code/data/imgedit/modified_instructions.csv')

	generator = ProgramGenerator(prompter=prompter, model_type=args.model_type, model_name=args.model, temperature=args.temperature, top_p=args.top_p, stop=args.stop, n=args.n, max_tokens=args.max_tokens, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty, port=args.port, timeout=args.timeout)

	if args.model_type in ['chat', 'completion']:
		encoding = tiktoken.encoding_for_model(args.model)
	else:
		encoding = None

	tot_input_tokens = 0
	tot_output_tokens = 0

	programs_ls = []

	print("Beginning Test for ImgEdit...")

	for i in range(len(test_df)):
		instruction = test_df.loc[i]['instruction']
		image_name = test_df.loc[i]['image_name']

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(i) + "\n")
			f.write("Instruction: " + instruction + "\n")

		_, prog, (cur_input_tokens, cur_output_tokens) = generator.generate(dict(instruction=instruction), tik_encoding=encoding)
		
		tot_input_tokens += cur_input_tokens
		tot_output_tokens += cur_output_tokens

		programs_ls.append([i, instruction, image_name, prog])
		programs_df = pd.DataFrame(programs_ls, columns=['Example', 'Instruction', 'Image', 'Program'])
		programs_df.to_csv(args.out_dir + "/generated_programs.csv", index=False)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Generated Program:\n" + prog  + "\n")

		print("Completed {}/{}".format(i, len(test_df)), end = '\r', flush = True)

	print("Total Input Tokens: " + str(tot_input_tokens))
	print("Total Output Tokens: " + str(tot_output_tokens))

def main(args):
	if args.data == 'nlvr':
		generate_nlvr(args)
	elif args.data == 'gqa':
		generate_gqa(args)
	elif args.data == 'knowtag':
		generate_knowtag(args)
	elif args.data == 'imgedit':
		generate_imgedit(args)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.data + "_" + args.prompt_type + "_" + args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	main(args)