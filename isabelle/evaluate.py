import pandas as pd
import sys
import os
import json
import argparse
import pdb
os.environ['ISABELLE_HOME'] = '/home/nlp/users/apatel79/Isabelle2022'
os.environ['PISA_PATH'] = '/home/nlp/users/apatel79/Portal-to-ISAbelle/src/main/python'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import dsp_utils

def build_parser():
	parser = argparse.ArgumentParser(description='Evaluate')

	parser.add_argument('-out_dir', type=str, default='results/', help='Output Directory')
	parser.add_argument('-progs_name', type=str, default='gpt-4-1', help='name of file with predicted programs')
	parser.add_argument('-port', type=int, default=9000, help='Port on which the evaluator is hosted')

	return parser

def add_sh(prog):
	lines = prog.split("\n")
	new_prog = ""
	for i in range(len(lines)):
		if " by " in lines[i]:
			new_prog = new_prog + lines[i].split(" by ")[0] + " sledgehammer"
		else:
			new_prog = new_prog + lines[i]
		if i < len(lines)-1:
			new_prog = new_prog + "\n"
	return new_prog

def main(args):
	checker = dsp_utils.Checker(
		working_dir = os.environ['ISABELLE_HOME'] + '/src/HOL/Examples',
		isa_path = os.environ['ISABELLE_HOME'],
		theory_file = os.environ['ISABELLE_HOME'] + '/src/HOL/Examples/Interactive.thy',
		port=args.port
	)

	pred_df = pd.read_csv(args.out_dir + args.progs_name + "/predictions.tsv", sep = '\t')

	tot = 0
	corr = 0

	exec_ls = []

	for i in range(len(pred_df)):
		pred_ex = pred_df.loc[i]
		name = pred_ex['Name']
		prob = json.loads(pred_ex['Problem'])
		pred = json.loads(pred_ex['Prediction'])

		result = checker.check(pred)
		pred_flag = result['success']
		reason = result['reason']
		comp_proof = result['theorem_and_proof']

		if not pred_flag:
			other_res = checker.check(add_sh(pred))
			if other_res['success']:
				pred_flag = True
				reason = other_res['reason']
				comp_proof = other_res['theorem_and_proof']

		tot += 1
		if pred_flag:
			corr += 1

		exec_ls.append([name, prob, pred, str(pred_flag), comp_proof])

		exec_df = pd.DataFrame(exec_ls, columns = ['Name', 'Problem', 'Prediction', 'Result', 'Complete Proof'])
		exec_df['Problem'] = exec_df['Problem'].apply(json.dumps)
		exec_df['Prediction'] = exec_df['Prediction'].apply(json.dumps)
		exec_df['Complete Proof'] = exec_df['Complete Proof'].apply(json.dumps)
		exec_df.to_csv(args.out_dir + args.progs_name + "/execution.tsv", sep = '\t', index = None)

		with open(args.out_dir + args.progs_name + "/exec_logs.txt", "a") as f:
			f.write("=========================================================================\n")
			f.write("Name: " + name + "\n\n")
			f.write("Prediction:\n" + pred + "\n\n")
			f.write("Success: " + str(pred_flag) + "\n\n")
			f.write("Reason: " + reason + "\n\n")
			f.write("Complete Proof:\n" + comp_proof + "\n")

		print("Completed {} / {}...".format(i, len(pred_df)), end = '\r', flush = True)

	acc = corr/tot
	print()
	print("Accuracy: ", acc)
	with open(args.out_dir + args.progs_name + "/exec_logs.txt", "a") as f:
		f.write("\n\nAccuracy: " + str(acc) + "\n\n")


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	main(args)
