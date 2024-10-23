import os
import argparse
import datetime
import random
import torch
import pandas as pd
import pdb
from PIL import Image, ImageDraw, ImageFont
from IPython.core.display import HTML
from functools import partial
import tiktoken

import sys
module_path = os.path.abspath(os.path.join('visprog_code'))
if module_path not in sys.path:
	sys.path.append(module_path)

from engine.utils import ProgramInterpreter
from engine.direct_interpreters import execute_program

def build_parser():
	parser = argparse.ArgumentParser(description='Evaluate LLMs on Vision')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-progs_name', type=str, default='nlvr_demos_gpt-3.5-turbo_0.7_2023-07-31-10:40:41_99', help='directory name containing generated programs')
	parser.add_argument('-out_dir', type=str, default='results/', help='Output Directory')
	parser.add_argument('-data', type=str, default='nlvr', help='data')
	
	return parser

def evaluate_gqa(args):
	interpreter = ProgramInterpreter(dataset=args.data)

	print("Beginning Evaluation for GQA-test...")
	print()

	test_df = pd.read_csv('visprog_code/data/gqa/test.tsv', sep='\t')
	test_progs_df = pd.read_csv(args.out_dir + "/test_generated_programs.csv")
	
	corr = 0.0
	tot = 0.0
	errors = 0

	for i in range(len(test_progs_df)):
		if i == 1039:
			continue
		idx = test_progs_df.loc[i]['Index']
		label = test_df[test_df['index'] == idx].reset_index(drop=True).loc[0]['answer']
		img_id = test_df[test_df['index'] == idx].reset_index(drop=True).loc[0]['imageId']
		generated_prog = test_progs_df.loc[i]['Program']
		question = test_df[test_df['index'] == idx].reset_index(drop=True).loc[0]['question']
		prog_question = test_progs_df.loc[i]['Question']
		assert question == prog_question

		img_name = str(img_id) + ".jpg"

		image = Image.open("visprog_code/data/gqa/images/" + img_name)
		image.thumbnail((640,640),Image.Resampling.LANCZOS)
		init_state = dict(
			IMAGE=image.convert('RGB')
		)

		with open(args.out_dir + "/test_exec_logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(i) + "\n")
			f.write("Question: " + question + "\n")
			f.write("Ground Truth Answer: " + str(label) + "\n")
			f.write("Generated Program:\n" + str(generated_prog)  + "\n")

		try:
			result_og_inter, prog_state, html_str = interpreter.execute(generated_prog,init_state,inspect=True)
		except:
			result_og_inter = "Error"
			prog_state = "Error"
			html_str = "Error"

		try:
			generated_prog = generated_prog.replace("EVAL", "EVAL2")
			result_cust_inter = execute_program(generated_prog, init_state)
		except:
			result_cust_inter = "Error"
			prog_state = "Error"
			html_str = "Error"

		if result_og_inter == "Error" and result_cust_inter == "Error":
			errors += 1

		correctness = False
		if str(result_og_inter).lower() == label.lower():
			correctness = True
		elif str(result_cust_inter).lower() == label.lower():
			correctness = True
		elif str(generated_prog).strip().lower() == label.lower():
			correctness = True

		tot += 1
		if correctness:
			corr += 1
		
		with open(args.out_dir + "/test_exec_logs.txt", "a") as f:
			f.write("Predicted Label using original interpreter: " + str(result_og_inter) + "\n")
			f.write("Predicted Label using custom interpreter: " + str(result_cust_inter) + "\n")
			f.write("Correctness: " + str(correctness) + "\n")

		print("Completed {}/{}".format(i, len(test_progs_df)), end = '\r', flush = True)

	print("Accuracy: " + str(corr/tot))
	print("Errors: " + str(errors))

	with open(args.out_dir + "/test_exec_logs.txt", "a") as f:
		f.write("============================================================================================\n")
		f.write("============================================================================================\n")
		f.write("Accuracy: " + str(corr/tot) + "\n")
		f.write("Errors: " + str(errors) + "\n")

def evaluate_nlvr(args):
	interpreter = ProgramInterpreter(dataset=args.data)
	
	test_df = pd.read_json('visprog_code/data/nlvr/nlvr2/data/test1.json', lines=True)

	progs_df = pd.read_csv(args.out_dir + "/generated_programs.csv")
	
	tp = 0.0
	fp = 0.0
	tn = 0.0
	fn = 0.0
	corr = 0.0
	tot = 0.0
	errors = 0

	for i in range(len(progs_df)):
		label = test_df.loc[i]['label']
		img_id = test_df.loc[i]['identifier']
		generated_prog = progs_df.loc[i]['Program']
		sentence = test_df.loc[i]['sentence']
		prog_sentence = progs_df.loc[i]['Sentence']
		assert sentence == prog_sentence

		left_img_name = img_id[:-2] + "-img0.png"
		right_img_name = img_id[:-2] + "-img1.png"

		left_image = Image.open("visprog_code/data/nlvr/nlvr2/images/" + left_img_name)
		left_image.thumbnail((640,640),Image.Resampling.LANCZOS)
		right_image = Image.open("visprog_code/data/nlvr/nlvr2/images/" + right_img_name)
		right_image.thumbnail((640,640),Image.Resampling.LANCZOS)
		init_state = dict(
			LEFT=left_image.convert('RGB'),
			RIGHT=right_image.convert('RGB'),
		)

		with open(args.out_dir + "/exec_logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(i) + "\n")
			f.write("Sentence: " + sentence + "\n")
			f.write("Ground Truth Answer: " + str(label) + "\n")
			f.write("Generated Program:\n" + str(generated_prog)  + "\n")

		try:
			result_og_inter, prog_state, html_str = interpreter.execute(generated_prog,init_state,inspect=True)
			if str(result_og_inter).lower() in ['yes', 'true']:
				result_og_inter = True
			if str(result_og_inter).lower() in ['no', 'false']:
				result_og_inter = False
		except:
			result_og_inter = "Error"
			prog_state = "Error"
			html_str = "Error"

		try:
			result_cust_inter = execute_program(generated_prog, init_state)
			if str(result_cust_inter).lower() in ['yes', 'true']:
				result_cust_inter = True
			if str(result_cust_inter).lower() in ['no', 'false']:
				result_cust_inter = False
		except:
			result_cust_inter = "Error"
			prog_state = "Error"
			html_str = "Error"

		if result_og_inter == "Error" and result_cust_inter == "Error":
			errors += 1

		correctness = False
		if str(result_og_inter) == label:
			correctness = True
			if label == "True":
				tp += 1
			else:
				tn += 1
		elif str(result_cust_inter) == label:
			correctness = True
			if label == "True":
				tp += 1
			else:
				tn += 1
		else:
			if label == "True":
				fn += 1
			else:
				fp += 1

		tot += 1
		if correctness:
			corr += 1
		
		with open(args.out_dir + "/exec_logs.txt", "a") as f:
			f.write("Predicted Label using original interpreter: " + str(result_og_inter) + "\n")
			f.write("Predicted Label using custom interpreter: " + str(result_cust_inter) + "\n")
			f.write("Correctness: " + str(correctness) + "\n")

		print("Completed {}/{}".format(i, len(progs_df)), end = '\r', flush = True)

	precision = tp / (tp + fp) if tp + fp != 0 else 0
	recall = tp / (tp + fn) if tp + fn != 0 else 0
	f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

	print("Accuracy: " + str(corr/tot))
	print("Precision: " + str(precision))
	print("Recall: " + str(recall))
	print("F1 Score: " + str(f1_score))
	print("Errors: " + str(errors))

	with open(args.out_dir + "/exec_logs.txt", "a") as f:
		f.write("============================================================================================\n")
		f.write("============================================================================================\n")
		f.write("Accuracy: " + str(corr/tot) + "\n")
		f.write("Precision: " + str(precision) + "\n")
		f.write("Recall: " + str(recall) + "\n")
		f.write("F1 Score: " + str(f1_score) + "\n")
		f.write("Errors: " + str(errors) + "\n")

def get_final_image(result, ques):
	img_width, img_height = result.size

	font = ImageFont.load_default()
	text_width, text_height = font.getsize(ques)

	new_height = img_height + text_height + 10  # Added 10 pixels as padding
	new_image = Image.new("RGB", (img_width, new_height), (255, 255, 255))  # White background
	new_image.paste(result, (0, 0))

	draw = ImageDraw.Draw(new_image)
	text_position = (img_width - text_width) // 2, img_height + 5  # Centered text with 5 pixels padding from the image
	draw.text(text_position, ques, font=font, fill=(0, 0, 0))  # Black text color

	return new_image

def calculate_IoU(pred_bbox, actual_bbox):
	# Determine the coordinates of the intersection rectangle
	x1 = max(pred_bbox[0], actual_bbox[0])
	y1 = max(pred_bbox[1], actual_bbox[1])
	x2 = min(pred_bbox[2], actual_bbox[2])
	y2 = min(pred_bbox[3], actual_bbox[3])

	# Compute the area of intersection rectangle
	inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

	# Compute the area of both the prediction and ground-truth rectangles
	pred_area = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
	actual_area = (actual_bbox[2] - actual_bbox[0] + 1) * (actual_bbox[3] - actual_bbox[1] + 1)

	# Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
	iou = inter_area / float(pred_area + actual_area - inter_area)

	return iou

def calculate_tagging(df, prediction_dict):
	TP = 0
	FP = 0
	FN = 0

	covered = []

	for label in prediction_dict:
		found = -1
		pred_bbox = prediction_dict[label]
		for i in range(len(df)):
			cur_labels = df.loc[i]['label_name'].lower().split("/")
			if label.lower() in cur_labels:
				found = i
				if i not in covered:
					covered.append(i)
				break
			for curl in cur_labels:
				if curl in label.lower() or label.lower() in curl:
					found = i
					if i not in covered:
						covered.append(i)
					break
		
		if found > -1:
			sub_df = df.iloc[found:found+1]
		else:
			FP += 1
			continue
		
		actual_bbox = sub_df[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values[0]
		
		actual_bbox = [actual_bbox[0], actual_bbox[1], 
					actual_bbox[0] + actual_bbox[2], 
					actual_bbox[1] + actual_bbox[3]]
		
		IoU = calculate_IoU(pred_bbox, actual_bbox)
		
		if IoU >= 0.5:
			TP += 1
		else:
			FP += 1

	FN = len([x for x in range(len(df)) if x not in covered])

	if TP + FP == 0:
		precision = 0
	else:
		precision = TP / (TP + FP)
	if TP + FN == 0:
		recall = 0
	else:
		recall = TP / (TP + FN)

	return precision, recall

def evaluate_knowtag(args):
	interpreter = ProgramInterpreter(dataset='okDet')
	
	test_df = pd.read_csv('visprog_code/data/knowtag/modified_instructions.csv')
	anno_df = pd.read_csv('visprog_code/data/knowtag/annotations.csv')

	progs_df = pd.read_csv(args.out_dir + "/generated_programs.csv")

	os.makedirs(args.out_dir + "/result_images", exist_ok=True)
	
	errors = 0

	precision = 0.0
	recall = 0.0

	for i in range(len(progs_df)):
		image_name = progs_df.loc[i]['Image']
		generated_prog = progs_df.loc[i]['Program']
		instruction = progs_df.loc[i]['Instruction']

		image_name = image_name + ".png"

		image = Image.open("visprog_code/data/knowtag/images/" + image_name)
		# image.thumbnail((640,640),Image.Resampling.LANCZOS)
		init_state = dict(
			IMAGE=image.convert('RGB')
		)

		with open(args.out_dir + "/exec_logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(i) + "\n")
			f.write("Instruction: " + instruction + "\n")
			f.write("Generated Program:\n" + str(generated_prog)  + "\n")

		try:
			generated_prog = generated_prog.replace("LOC", "LOC2")
			result_cust_inter = execute_program(generated_prog, init_state)
			result = result_cust_inter

			eval_prog = generated_prog.replace("TAG", "TAG_EVAL")
			eval_result1 = execute_program(eval_prog, init_state)
			eval_result2 = execute_program(eval_prog, init_state)
			eval_result3 = execute_program(eval_prog, init_state)
			
			gold_df = anno_df[anno_df['image_name'] == image_name].reset_index(drop=True)

			cur_precision1, cur_recall1 = calculate_tagging(gold_df, eval_result1)
			cur_precision2, cur_recall2 = calculate_tagging(gold_df, eval_result2)
			cur_precision3, cur_recall3 = calculate_tagging(gold_df, eval_result3)
			precision += max(cur_precision1, cur_precision2, cur_precision3)
			recall += max(cur_recall1, cur_recall2, cur_recall3)

		except:
			result_cust_inter = "Error"
			prog_state = "Error"
			html_str = "Error"

		if result_cust_inter == "Error":
			errors += 1
			with open(args.out_dir + "/exec_logs.txt", "a") as f:
				f.write("FAILURE\n")
		else:
			final_result = get_final_image(result, instruction)
			final_result.save(args.out_dir + "/result_images/" + str(i) + ".png", 'PNG')

			with open(args.out_dir + "/exec_logs.txt", "a") as f:
				f.write("SUCCESS\n")
				f.write("gold_df: " + str(gold_df) + "\n")
				f.write("pred_tags: " + str([eval_result1, eval_result2, eval_result3]) + "\n")
				f.write("Precision: " + str(max(cur_precision1, cur_precision2, cur_precision3)) + "\n")
				f.write("Recall: " + str(max(cur_recall1, cur_recall2, cur_recall3)) + "\n")

		print("Completed {}/{}".format(i, len(progs_df)), end = '\r', flush = True)

	avg_precision = precision / len(progs_df)
	avg_recall = recall / len(progs_df)
	f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

	print("Average Precision: " + str(avg_precision))
	print("Average Recall: " + str(avg_recall))
	print("F1 Score: " + str(f1))
	print("Errors: " + str(errors))

	with open(args.out_dir + "/exec_logs.txt", "a") as f:
		f.write("============================================================================================\n")
		f.write("============================================================================================\n")
		f.write("Average Precision: " + str(avg_precision) + "\n")
		f.write("Average Recall: " + str(avg_recall) + "\n")
		f.write("F1 Score: " + str(f1) + "\n")
		f.write("Errors: " + str(errors) + "\n")

def evaluate_imgedit(args):
	interpreter = ProgramInterpreter(dataset='imageEdit')
	
	test_df = pd.read_csv('visprog_code/data/imgedit/modified_instructions.csv')

	progs_df = pd.read_csv(args.out_dir + "/generated_programs.csv")

	os.makedirs(args.out_dir + "/result_images", exist_ok=True)
	
	errors = 0

	for i in range(len(progs_df)):
		image_name = progs_df.loc[i]['Image']
		generated_prog = progs_df.loc[i]['Program']
		instruction = progs_df.loc[i]['Instruction']

		image_name = image_name + ".png"

		image = Image.open("visprog_code/data/imgedit/images/" + image_name)
		image.thumbnail((640,640),Image.Resampling.LANCZOS)
		init_state = dict(
			IMAGE=image.convert('RGB')
		)

		with open(args.out_dir + "/exec_logs.txt", "a") as f:
			f.write("----------------------------------------------------------------------\n")
			f.write("Example: " + str(i) + "\n")
			f.write("Instruction: " + instruction + "\n")
			f.write("Generated Program:\n" + str(generated_prog)  + "\n")

		try:
			result_og_inter, prog_state, html_str = interpreter.execute(generated_prog,init_state,inspect=True)
			result = result_og_inter
		except:
			result_og_inter = "Error"
			prog_state = "Error"
			html_str = "Error"

			try:
				result_cust_inter = execute_program(generated_prog, init_state)
				result = result_cust_inter
			except:
				result_cust_inter = "Error"
				prog_state = "Error"
				html_str = "Error"

		if result_og_inter == "Error" and result_cust_inter == "Error":
			errors += 1
			with open(args.out_dir + "/exec_logs.txt", "a") as f:
				f.write("FAILURE\n")
		else:
			final_result = get_final_image(result, instruction)
			final_result.save(args.out_dir + "/result_images/" + str(i) + ".png", 'PNG')

			with open(args.out_dir + "/exec_logs.txt", "a") as f:
				f.write("SUCCESS\n")

		print("Completed {}/{}".format(i, len(progs_df)), end = '\r', flush = True)

	print("Errors: " + str(errors))

	with open(args.out_dir + "/exec_logs.txt", "a") as f:
		f.write("============================================================================================\n")
		f.write("============================================================================================\n")
		f.write("Errors: " + str(errors) + "\n")

def main(args):
	if args.data == 'nlvr':
		evaluate_nlvr(args)
	elif args.data == 'gqa':
		evaluate_gqa(args)
	elif args.data == 'knowtag':
		evaluate_knowtag(args)
	elif args.data == 'imgedit':
		evaluate_imgedit(args)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	
	args.run_name = args.progs_name

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)
	
	device = torch.device("cuda:0")

	args.device = device

	main(args)