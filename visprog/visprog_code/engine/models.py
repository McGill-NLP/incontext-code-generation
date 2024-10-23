import os
import time
import argparse
import uuid
from typing import List
import pdb

import openai
from openai import OpenAI
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.wait import wait_random_exponential
import tiktoken

from vllm import SamplingParams, RequestOutput

@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.RateLimitError,
				openai.APIConnectionError,
				openai.InternalServerError,
				openai.APITimeoutError
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_completion_response(client, engine, prompt, sys_prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty, best_of, logprobs=1, echo=False):
	fin_prompt = sys_prompt + "\n\n" + prompt
	return client.completions.create(
		model=engine,
		prompt=fin_prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty,
		best_of=best_of,
		logprobs=logprobs,
		echo=echo
	)

@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.RateLimitError,
				openai.APIConnectionError,
				openai.InternalServerError,
				openai.APITimeoutError
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_chat_response(client, engine, prompt, sys_prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty):
	if "Mixtral-8x22B" in engine or "gemma" in engine:
		return client.chat.completions.create(
			model=engine,
			messages = [
				{"role": "user", "content": sys_prompt + "\n\n" + prompt}
			],
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			n=n,
			stop=stop,
			presence_penalty=presence_penalty,
			frequency_penalty=frequency_penalty
		)
	return client.chat.completions.create(
		model=engine,
		messages = [
			{"role": "system", "content": sys_prompt},
			{"role": "user", "content": prompt}
		],
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty
	)


@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.RateLimitError,
				openai.APIConnectionError,
				openai.InternalServerError,
				openai.APITimeoutError
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_strawberry_response(client, engine, prompt):
	return client.chat.completions.create(
		model=engine,
		messages = [
			{"role": "user", "content": prompt}
		]
	)


class LargeLanguageModel():
	def __init__(self, model_type, model, top_p, presence_penalty, frequency_penalty, port=8080, timeout=1000000):
		self.model_type = model_type
		self.engine = model
		self.top_p = top_p
		self.presence_penalty = presence_penalty
		self.frequency_penalty = frequency_penalty

		if self.model_type in ['vllm']:
			openai_api_key = "EMPTY"
			openai_api_base = "http://localhost:8000/v1"
			self.client = OpenAI(
				api_key=openai_api_key,
				base_url=openai_api_base,
			)
		elif self.model_type in ['chat', 'completion', 'o1']:
			self.client = OpenAI(
				api_key=openai.api_key,
			)

	def predict(self, prompt, sys_prompt, max_tokens, temperature=0.0, n=1, stop = []):
		if self.model_type == "completion":
			response = _get_completion_response(
				client=self.client,
				engine=self.engine,
				prompt=prompt,
				sys_prompt=sys_prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				top_p=self.top_p,
				n=n,
				stop=stop,
				presence_penalty=self.presence_penalty,
				frequency_penalty=self.frequency_penalty,
				best_of=n+1,
				echo=False
			)
			response = response["choices"][0]['text'].lstrip('\n').rstrip('\n')
		elif self.model_type in ["chat", "vllm"]:
			# pdb.set_trace()
			response = "error"
			cur_max_tokens = max_tokens
			while(cur_max_tokens > 0):
				try:
					response = _get_chat_response(
						client=self.client,
						engine=self.engine,
						prompt=prompt, 
						sys_prompt=sys_prompt,
						max_tokens=cur_max_tokens,
						temperature=temperature,
						top_p=self.top_p,
						n=n,
						stop=stop,
						presence_penalty=self.presence_penalty,
						frequency_penalty=self.frequency_penalty
					)
					response = response.choices[0].message.content.lstrip('\n').rstrip('\n')
					break
				except openai.BadRequestError:
					cur_max_tokens = cur_max_tokens - 2000
		elif self.model_type in ["o1"]:
			response = _get_strawberry_response(
				client=self.client,
				engine=self.engine,
				prompt=prompt
			)
			response = response.choices[0].message.content.lstrip('\n').rstrip('\n')
			# pdb.set_trace()
		return response