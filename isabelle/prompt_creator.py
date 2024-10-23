import json
import os
import random
import pdb

path = "prompt_examples"

prompt_tags = []

examples = {}

for filename in os.listdir(path):
	filepath = os.path.join(path, filename)
	with open(filepath) as f:
		ex = json.load(f)
		tag = ex['tag']
		prompt_tags.append(tag)
		cat = ex['category']
		prompt = ex['prompt']
		if cat in examples:
			examples[cat].append(prompt)
		else:
			examples[cat] = [prompt]

categories = {
	"aime": "algebra", 
	"algebra": "algebra", 
	"amc": "algebra", 
	"induction": "algebra", 
	"mathd_algebra": "algebra", 
	"mathd_numbertheory": "number_theory", 
	"numbertheory": "number_theory",
	"imo": "number_theory"
}

def get_prompt(problem, prompt_type='dsp', num_ex=8, keyword_alias=None, alias_keyword=None):
	name = problem['problem_name']
	if name in prompt_tags:
		return "PROMPT_EXAMPLE", "PROMPT_EXAMPLE"

	if prompt_type == 'zero-shot':
		sys_prompt = "You are an expert theorem prover in Isabelle. Provide the formal proof sketch for the theorem stated. You must only generate the isabelle proof and nothing else."
		prompt = "Translate the informal solution into a sketch of the formal Isabelle proof. Add `sledgehammer` in the sketch whenever possible. `sledgehammer` will be used to call the automated Sledgehammer prover.\n\n"
		
		prompt += "Informal:\n(*### Problem\n\n"
		prompt += json.loads(problem["informal_statement"]) + "\n\n"
		prompt += "### Solution\n\n"
		try:
			prompt += json.loads(problem["informal_proof"]) + "*)\n\n"
		except:
			prompt += "*)\n\n"
		prompt += "Formal:\n"
		prompt += json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem") + "\nproof -"
	elif prompt_type == 'dsp':
		sys_prompt = "You are an expert theorem prover in Isabelle. Provide the formal proof sketch for the final theorem stated. You must only generate the isabelle proof and nothing else."
		cur_cat = "None"
		for cat in categories:
			if cat == name[:len(cat)]:
				cur_cat = cat
				break
		categ = categories[cur_cat]
		if categ == "algebra":
			selected_ex = random.choices(examples[categ], k=num_ex)
		else:
			selected_ex = examples[categ]
		prompt = "Translate the informal solution into a sketch of the formal Isabelle proof. Add `sledgehammer` in the sketch whenever possible. `sledgehammer` will be used to call the automated Sledgehammer prover. Here are some examples:\n\n"
		
		for ex in selected_ex:
			prompt += (ex + "\n\n")
		prompt += "Informal:\n(*### Problem\n\n"
		prompt += json.loads(problem["informal_statement"]) + "\n\n"
		prompt += "### Solution\n\n"
		try:
			prompt += json.loads(problem["informal_proof"]) + "*)\n\n"
		except:
			prompt += "*)\n\n"
		prompt += "Formal:\n"
		prompt += json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem") + "\nproof -"
	elif prompt_type == 'alias_dsp':
		sys_prompt = "You are an expert theorem prover. Provide the formal proof sketch for the final theorem stated. You must only generate the proof in the corresponding programming language and nothing else. The generated proof must not introduce any new keywords apart from the ones already used in the demonstrations provided."
		cur_cat = "None"
		for cat in categories:
			if cat == name[:len(cat)]:
				cur_cat = cat
				break
		categ = categories[cur_cat]
		if categ == "algebra":
			selected_ex = random.choices(examples[categ], k=num_ex)
		else:
			selected_ex = examples[categ]
		prompt = "Translate the informal solution into a sketch of the formal proof in the corresponding programming language. The formal proof must only include keywords illustrated in the examples below. Add `vwxy` in the sketch whenever possible. `vwxy` will be used to call the automated Sledgehammer prover. Here are some examples:\n\n"
		
		for ex in selected_ex:
			temp_ex = ex
			for keyword in keyword_alias:
				temp_ex = temp_ex.replace(keyword, keyword_alias[keyword])
			prompt += (temp_ex + "\n\n")
		prompt += "Informal:\n(*### Problem\n\n"
		prompt += json.loads(problem["informal_statement"]) + "\n\n"
		prompt += "### Solution\n\n"
		try:
			prompt += json.loads(problem["informal_proof"]) + "*)\n\n"
		except:
			prompt += "*)\n\n"
		prompt += "Formal:\n"
		formal_stmt = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		for keyword in keyword_alias:
			formal_stmt = formal_stmt.replace(keyword, keyword_alias[keyword])
		prompt += formal_stmt + "\nproof -"
	elif prompt_type == 'alias_isabelle_desc':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc.txt") as f:
			prompt = f.read()
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		for keyword in keyword_alias:
			temp_prob = temp_prob.replace(keyword, keyword_alias[keyword])
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'alias_isabelle_desc_examples_2':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_examples_2.txt") as f:
			prompt = f.read()
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		for keyword in keyword_alias:
			temp_prob = temp_prob.replace(keyword, keyword_alias[keyword])
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'alias_isabelle_desc_examples_5':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_examples_5.txt") as f:
			prompt = f.read()
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		for keyword in keyword_alias:
			temp_prob = temp_prob.replace(keyword, keyword_alias[keyword])
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'alias_isabelle_desc_examples_7':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_examples_7.txt") as f:
			prompt = f.read()
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		for keyword in keyword_alias:
			temp_prob = temp_prob.replace(keyword, keyword_alias[keyword])
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'alias_isabelle_desc_examples_10':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_examples_10.txt") as f:
			prompt = f.read()
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		for keyword in keyword_alias:
			temp_prob = temp_prob.replace(keyword, keyword_alias[keyword])
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'isabelle_desc':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc.txt") as f:
			prompt = f.read()
		for alias in alias_keyword:
			prompt = prompt.replace(alias, alias_keyword[alias])
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'isabelle_desc_os':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_os.txt") as f:
			prompt = f.read()
		for alias in alias_keyword:
			prompt = prompt.replace(alias, alias_keyword[alias])
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
		prompt += "\n\nNow generate the formal proof for the above theorem using only the theoremPL language and the keywords 'have', 'using', 'sledgehammer', 'then', 'show', '?thesis', 'qed':"
	elif prompt_type == 'isabelle_desc_examples':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_examples.txt") as f:
			prompt = f.read()
		for alias in alias_keyword:
			prompt = prompt.replace(alias, alias_keyword[alias])
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'isabelle_desc_examples_5':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_examples_5.txt") as f:
			prompt = f.read()
		for alias in alias_keyword:
			prompt = prompt.replace(alias, alias_keyword[alias])
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
	elif prompt_type == 'isabelle_desc_examples_5_os':
		sys_prompt = "You are an expert theorem prover and efficient learner of programming languages. Generate only the program to the best of your ability. Do not generate anything that is not in the programming language specified. Do not refuse or provide any explanation."
		with open("isabelle_desc_examples_5_os.txt") as f:
			prompt = f.read()
		for alias in alias_keyword:
			prompt = prompt.replace(alias, alias_keyword[alias])
		prompt += "\n\n"
		temp_prob = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem")
		prompt += temp_prob
		prompt += "\n\nI am providing an informal proof sketch for your reference:\n" + json.loads(problem["informal_proof"])
		prompt += "\n\nNow generate the formal proof for the above theorem using only the theoremPL language and the keywords 'have', 'using', 'sledgehammer', 'then', 'show', '?thesis', 'qed':"
	else:
		sys_prompt = "You are an expert theorem prover in Isabelle. Provide the formal proof sketch for the final theorem stated. You must only generate the isabelle proof and nothing else."
		prompt = json.loads(problem['formal_statement']).replace("theorem " + name + ":", "theorem") + "\nproof -"

	return prompt, sys_prompt