import os
from PIL import Image
import openai
import numpy as np
import copy
import pdb

from .step_interpreters import register_step_interpreters, parse_step
from .models import LargeLanguageModel


class Program:
	def __init__(self,prog_str,init_state=None):
		self.prog_str = prog_str
		self.state = init_state if init_state is not None else dict()
		self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
	def __init__(self,dataset='nlvr'):
		self.step_interpreters = register_step_interpreters(dataset) # dictionary of module names mapped to module classes
		openai.api_key = os.getenv("MODELSEC")

	def execute_step(self,prog_step,inspect):
		step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
		print(step_name)
		return self.step_interpreters[step_name].execute(prog_step,inspect)

	def execute(self,prog,init_state,inspect=False):
		if isinstance(prog,str):
			prog = Program(prog,init_state)
		else:
			assert(isinstance(prog,Program))

		prog_steps = [Program(instruction,init_state=prog.state) \
			for instruction in prog.instructions]

		html_str = '<hr>'
		for prog_step in prog_steps:
			if inspect:
				step_output, step_html = self.execute_step(prog_step,inspect)
				html_str += step_html + '<hr>'
			else:
				step_output = self.execute_step(prog_step,inspect)

		if inspect:
			return step_output, prog.state, html_str

		return step_output, prog.state


class ProgramGenerator():
	def __init__(self,prompter,model_type='chat',model_name='gpt-3.5-turbo',temperature=0.7,top_p=1.0,stop=[],n=1,max_tokens=512,presence_penalty=0.0,frequency_penalty=0.0,prob_agg='mean',port=8080,timeout=1000000):
		openai.api_key = os.getenv("OPENAI_KEY")
		self.prompter = prompter
		self.temperature = temperature
		self.top_p = top_p
		self.stop=stop
		self.n = n
		self.max_tokens = max_tokens
		self.prob_agg = prob_agg
		self.model = LargeLanguageModel(
			model_type=model_type,
			model=model_name,
			top_p=top_p,
			presence_penalty=presence_penalty,
			frequency_penalty=frequency_penalty,
			port=port,
			timeout=timeout
		)

	def compute_prob(self,response):
		eos = '<|endoftext|>'
		for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
			if token==eos:
				break

		if self.prob_agg=='mean':
			agg_fn = np.mean
		elif self.prob_agg=='sum':
			agg_fn = np.sum
		else:
			raise NotImplementedError

		return np.exp(agg_fn(
			response.choices[0]['logprobs']['token_logprobs'][:i]))

	def generate(self,inputs, tik_encoding=None):
		# pdb.set_trace()
		sys_prompt = "You are a Vision Programmer. Given a set of functions, generate only the corresponding program calling those functions for answering the given statement. The program will have more than one line. Each line of the program must call exactly one of the given functions. At the end, assign the answer to a variable called FINAL_ANSWER by calling the RESULT() function."
		# sys_prompt = "You are a QA bot. Do not refrain from answering. Do not refuse. Just take a guess and answer."
		prog = self.model.predict(prompt=self.prompter(inputs),sys_prompt=sys_prompt,max_tokens=self.max_tokens,temperature=self.temperature,n=self.n,stop=self.stop)
		num_input_tokens = 0
		num_output_tokens = 0
		if tik_encoding is not None:
			num_input_tokens = len(tik_encoding.encode(self.prompter(inputs)))
			num_output_tokens = len(tik_encoding.encode(prog))

		# prob = self.compute_prob(response)
		prob = 1.0
		# prog = response.choices[0]['text'].lstrip('\n').rstrip('\n')
		return prob, prog, (num_input_tokens, num_output_tokens)
	