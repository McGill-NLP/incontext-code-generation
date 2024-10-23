import random

NLVR_CURATED_EXAMPLES=[
"""Statement: An image shows one bare hand with the thumb on the right holding up a belly-first, head-up crab, with water in the background.
Program:
ANSWER0=VQA(image=LEFT,question="Does the image shows one bare hand with the thumb on the right holding a crab?")
ANSWER1=VQA(image=RIGHT,question="Does the image shows one bare hand with the thumb on the right holding a crab?")
ANSWER2=VQA(image=LEFT,question="Is the crab belly-first and head-ups?")
ANSWER3=VQA(image=RIGHT,question="Is the crab belly-first and head-ups?")
ANSWER4=VQA(image=LEFT,question="Is there water in the background?")
ANSWER5=VQA(image=RIGHT,question="Is there water in the background?")
ANSWER6=EVAL(expr="'{ANSWER0}' and '{ANSWER2}' and '{ANSWER4}'")
ANSWER7=EVAL(expr="'{ANSWER1}' and '{ANSWER3}' and '{ANSWER5}'")
ANSWER8=EVAL(expr="'{ANSWER6}' xor '{ANSWER7}'")
FINAL_ANSWER=RESULT(var=ANSWER8)
""",
"""Statement: There is a red convertible in one image.
Program:
ANSWER0=VQA(image=LEFT,question="Is there a red convertible in the image?")
ANSWER1=VQA(image=RIGHT,question="Is there a red convertible in the image?")
ANSWER2=EVAL(expr="'{ANSWER0}' xor '{ANSWER1}'")
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: One dog is laying down.
Program:
ANSWER0=VQA(image=LEFT,question="How many dogs are laying down?")
ANSWER1=VQA(image=RIGHT,question="How many dogs are laying down?")
ANSWER2=EVAL(expr="'{ANSWER0}' + '{ANSWER1}' == 1")
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: There are two blue and yellow birds
Program:
ANSWER0=VQA(image=LEFT,question="How many blue and yellow birds are in the image?")
ANSWER1=VQA(image=RIGHT,question="How many blue and yellow birds are in the image?")
ANSWER2=EVAL(expr="'{ANSWER0}' + '{ANSWER1}' == 2")
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: A single wolf is howling and silhouetted by the moon in one of the images.
Program:
ANSWER0=VQA(image=LEFT,question="How many wolves are in the image?")
ANSWER1=VQA(image=RIGHT,question="How many wolves are in the image?")
ANSWER2=VQA(image=LEFT,question="Is the wolf howling and silhouetted by the moon?")
ANSWER3=VQA(image=RIGHT,question="Is the wolf howling and silhouetted by the moon?")
ANSWER4=EVAL(expr="'{ANSWER0}' == 1 and '{ANSWER2}'")
ANSWER5=EVAL(expr="'{ANSWER1}' == 1 and '{ANSWER3}'")
ANSWER6=EVAL(expr="'{ANSWER4}' xor '{ANSWER5}'")
FINAL_ANSWER=RESULT(var=ANSWER6)
""",
"""Statement: One of the two images has a bag with the characters from Disney"s Frozen on it.
Program:
ANSWER0=VQA(image=LEFT,question="Does the image have a bag with the characters from Disney's Frozen on it?")
ANSWER1=VQA(Image=RIGHT,question="Does the image have a bag with the characters from Disney's Frozen on it?")
ANSWER2=EVAL(expr="'{ANSWER0}' xor '{ANSWER1}'")
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Statement: there are at least seven wine bottles in the image on the left
Program:
ANSWER0=VQA(image=LEFT,question="How many wine bottles are in the image?")
ANSWER1=EVAL(expr="'{ANSWER0}' >= 7")
FINAL_ANSWER=RESULT(var=ANSWER1)
""",
"""Statement: An image shows broccoli growing in soil, with leaves surrounding the florets.
Program:
ANSWER0=VQA(image=LEFT,question="Does the image show broccoli growing in soil?")
ANSWER1=VQA(image=RIGHT,question="Does the image show broccoli growing in soil?")
ANSWER2=VQA(image=LEFT,question="Are leaves surrounding the floret?")
ANSWER3=VQA(image=RIGHT,question="Are leaves surrounding the floret?")
ANSWER4=EVAL(expr="'{ANSWER0}' and '{ANSWER2}'")
ANSWER5=EVAL(expr="'{ANSWER1}' and '{ANSWER3}'")
ANSWER6=EVAL(expr="'{ANSWER4}' xor '{ANSWER5}'")
FINAL_ANSWER=RESULT(var=ANSWER6)
""",
"""Statement: An image shows exactly two seals in direct contact, posed face to face.
Program:
ANSWER0=VQA(image=LEFT,question="How many seals are in the image?")
ANSWER1=VQA(image=RIGHT,question="How many seals are in the image?")
ANSWER2=VQA(image=LEFT,question="Are the seals in direct contact?")
ANSWER3=VQA(image=RIGHT,question="Are the seals in direct contact?")
ANSWER4=VQA(image=LEFT,question="Are the seals posed face to face?")
ANSWER5=VQA(image=RIGHT,question="Are the seals posed face to face?")
ANSWER6=EVAL(expr="'{ANSWER0}' == 2 and '{ANSWER2}' and '{ANSWER4}'")
ANSWER7=EVAL(expr="'{ANSWER1}' == 2 and '{ANSWER3}' and '{ANSWER5}'")
ANSWER8=EVAL(expr="'{ANSWER6}' xor '{ANSWER7}'")
FINAL_ANSWER=RESULT(var=ANSWER8)
""",
"""Statement: There is at least two parrots in the right image.
Program:
ANSWER0=VQA(image=RIGHT,question="How many parrots are in the image?")
ANSWER1=EVAL(expr="'{ANSWER0}' >= 2")
FINAL_ANSWER=RESULT(var=ANSWER1)
""",
"""Statement: In the image on the right, four people are riding in one canoe.
Program:
ANSWER0=VQA(image=RIGHT,question="Are there four people riding in one canoe?")
ANSWER1=EVAL(expr="'{ANSWER0}'")
FINAL_ANSWER=RESULT(var=ANSWER1)
""",
"""Statement: There are two wolves in each image.
Program:
ANSWER0=VQA(image=LEFT,question="How many wolves are in the image?")
ANSWER1=VQA(image=RIGHT,question="How many wolves are in the image?")
ANSWER2=EVAL(expr="'{ANSWER0}' == 2 and '{ANSWER1}' == 2")
FINAL_ANSWER=RESULT(var=ANSWER2)
"""
]

prompt_dict = {
"docstring_all": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def LOC(image: IMAGE, object: str) -> BOX:
	# Returns the bounding boxes of the object in the image. The output of this function must be passed as an argument to one of the cropping functions or the COUNT function or both.

def COUNT(box: BOX) -> int:
	# Returns the number of objects detected. The input can only be an output of the LOC function.

def CROP(image: IMAGE, box: BOX) -> IMAGE:
	# Returns the cropped image corresponding to the bounding boxes.

def CROP_LEFTOF(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is on the left of the bounding box.

def CROP_RIGHTOF(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is on the right of the bounding box.

def CROP_ABOVE(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is above the bounding box.

def CROP_BELOW(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is below the bounding box.

def FACEDET(image: IMAGE) -> OBJ:
	# Detects the faces in the image and provides bounding boxes. The output of this function must be passed as the object argument to the SELECT function.
	
def SEG(image: IMAGE) -> OBJ:
    # Segments the image and provides bounding boxes. The output of this function must be passed as the object argument to the SELECT function.

def SELECT(image: IMAGE, object: OBJ, query: str, category: str) -> OBJ:
    # Selects the object from the image that needs to be edited. The output of this function must be passed as the object argument to the EMOJI, COLORPOP, BGBLUR, and REPLACE functions.

def EMOJI(image: IMAGE, object: OBJ, emoji: str) -> IMAGE:
    # Replaces the object in the image with the emoji and returns the edited image.
	
def COLORPOP(image: IMAGE, object: OBJ) -> IMAGE:
    # Creates a color pop of the objects in the image and returns the edited image.
	
def BGBLUR(image: IMAGE, object: OBJ) -> IMAGE:
    # Blurs the background of the image and returns the edited image.
	
def REPLACE(image: IMAGE, object: OBJ, prompt: str) -> IMAGE:
    # Replaces the object in the image with the object mentioned by the prompt and returns the edited image.

def LIST(query: str, max: int) -> List:
	# Returns a list of objects retrieved according to the query. The output of this function must be passed as the categories argument to the CLASSIFY function.

def CLASSIFY(image: IMAGE, object: OBJ, categories: List) -> OBJ:
	# Classifies the objects in the image according to the categories. Takes lists of object regions and categories and assigns one of the categories to each region. The output of this function must be passed as the object argument to the TAG function.

def TAG(image: IMAGE, object: OBJ) -> IMAGE:
	# Tags the objects in the image. Takes a list of object regions and tags each region with the corresponding category. The output of this function must be passed as the var argument to the RESULT function.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. The final answer must be either "yes" or "no". If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
""",

"docstring_basic": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. The final answer must be either "yes" or "no". If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
""",

"docstring_basic_os": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program to obtain the answer to the question. 
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. The program must assign the final answer to a variable called FINAL_ANSWER in the last line.
5. The final answer must be either "yes" or "no".
6. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images.
""",

"docstring_basic_old": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. Perform boolean checks by passing input to the EVAL() function. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument. The input argument when calling EVAL() must be a string created using local variables and concatenation using + operators.
""",

"docstring_indi_1ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.
Example:
ANSWER1=VQA(image=RIGHT,question=<subquestion>)

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.
Example:
ANSWER5=EVAL(expr='{ANSWER1} <boolean operator> {ANSWER3}')

def RESULT(var: str) -> str:
	# Returns the final answer.
Example:
FINAL_ANSWER=RESULT(var=ANSWER6)

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program that returns the answer to the question. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. Perform boolean checks by passing input to the EVAL() function. 
""",

"docstring_1ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

One example of a program using the above functions and variables:

ANSWER0=VQA(image=LEFT,question=<subquestion>)
ANSWER1=VQA(image=RIGHT,question=<subquestion>)
ANSWER2=VQA(image=LEFT,question=<subquestion>)
ANSWER3=VQA(image=RIGHT,question=<subquestion>)
ANSWER4=EVAL(expr='{ANSWER0} <boolean operator> {ANSWER2}')
ANSWER5=EVAL(expr='{ANSWER1} <boolean operator> {ANSWER3}')
ANSWER6=EVAL(expr='{ANSWER4} <boolean operator> {ANSWER5}')
FINAL_ANSWER=RESULT(var=ANSWER6)

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program that returns the answer to the question. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. Perform boolean checks by passing input to the EVAL() function. 
""",

"docstring_3ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

Some examples of programs using the above functions and variables:
example 1:
ANSWER0=VQA(image=LEFT,question=<subquestion>)
ANSWER1=VQA(image=RIGHT,question=<subquestion>)
ANSWER2=EVAL(expr='{ANSWER0} <boolean operator> {ANSWER2}')
FINAL_ANSWER=RESULT(var=ANSWER2)

example 2:
ANSWER0=VQA(image=RIGHT,question=<subquestion>)
ANSWER1=EVAL(expr='{ANSWER0} == <number>')
FINAL_ANSWER=RESULT(var=ANSWER1)

example 3:
ANSWER0=VQA(image=LEFT,question=<subquestion>)
ANSWER1=VQA(image=RIGHT,question=<subquestion>)
ANSWER2=VQA(image=LEFT,question=<subquestion>)
ANSWER3=VQA(image=RIGHT,question=<subquestion>)
ANSWER4=EVAL(expr='{ANSWER0} <boolean operator> {ANSWER2}')
ANSWER5=EVAL(expr='{ANSWER1} <boolean operator> {ANSWER3}')
ANSWER6=EVAL(expr='{ANSWER4} <boolean operator> {ANSWER5}')
FINAL_ANSWER=RESULT(var=ANSWER6)

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program that returns the answer to the question. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. Perform boolean checks by passing input to the EVAL() function. 
""",

"docstring_6ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

Some examples of programs using the above functions and variables:
example 1:
ANSWER0=VQA(image=LEFT,question=<subquestion>)
ANSWER1=VQA(image=RIGHT,question=<subquestion>)
ANSWER2=EVAL(expr='{ANSWER0} <boolean operator> {ANSWER2}')
FINAL_ANSWER=RESULT(var=ANSWER2)

example 2:
ANSWER0=VQA(image=RIGHT,question=<subquestion>)
ANSWER1=EVAL(expr='{ANSWER0} == <number>')
FINAL_ANSWER=RESULT(var=ANSWER1)

example 3:
ANSWER0=VQA(image=LEFT,question=<subquestion>)
ANSWER1=VQA(image=RIGHT,question=<subquestion>)
ANSWER2=VQA(image=LEFT,question=<subquestion>)
ANSWER3=VQA(image=RIGHT,question=<subquestion>)
ANSWER4=EVAL(expr='{ANSWER0} <boolean operator> {ANSWER2}')
ANSWER5=EVAL(expr='{ANSWER1} <boolean operator> {ANSWER3}')
ANSWER6=EVAL(expr='{ANSWER4} <boolean operator> {ANSWER5}')
FINAL_ANSWER=RESULT(var=ANSWER6)

example 4:
ANSWER0=VQA(image=LEFT,question=<subquestion>)
ANSWER1=VQA(image=RIGHT,question=<subquestion>)
ANSWER2=VQA(image=LEFT,question=<subquestion>)
ANSWER3=VQA(image=RIGHT,question=<subquestion>)
ANSWER4=VQA(image=LEFT,question=<subquestion>)
ANSWER5=VQA(image=RIGHT,question=<subquestion>)
ANSWER6=EVAL(expr='{ANSWER0} <boolean operator> {ANSWER2} <boolean operator> {ANSWER4}')
ANSWER7=EVAL(expr='{ANSWER1} <boolean operator> {ANSWER3} <boolean operator> {ANSWER5}')
ANSWER8=EVAL(expr='{ANSWER6} <boolean operator> {ANSWER7}')
FINAL_ANSWER=RESULT(var=ANSWER8)

example 5:
ANSWER0=VQA(image=RIGHT,question=<subquestion>)
ANSWER1=EVAL(expr='{ANSWER0} >= <number>')
FINAL_ANSWER=RESULT(var=ANSWER1)

example 6:
ANSWER0=VQA(image=LEFT,question=<subquestion>)
ANSWER1=VQA(image=RIGHT,question=<subquestion>)
ANSWER2=EVAL(expr='{ANSWER0} + {ANSWER1} == <number>')
FINAL_ANSWER=RESULT(var=ANSWER2)

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program that returns the answer to the question. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. Perform boolean checks by passing input to the EVAL() function. 
""",

"imple_basic": """You are given the following code:

from transformers import (ViltProcessor, ViltForQuestionAnswering, 
	OwlViTProcessor, OwlViTForObjectDetection,
	MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
	CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)
vqa_model.eval()

def VQA(image: IMAGE, question: str) -> str:
	encoding = vqa_processor(image, question, return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = vqa_model.generate(**encoding)

	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
		
	return ans_op

def RESULT(var: str) -> str:
	return var

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions, VQA(), and RESULT(), and variables LEFT and RIGHT, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. The final answer must be either "yes" or "no". If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
""",

"imple_basic_os": """You are given the following code:

from transformers import (ViltProcessor, ViltForQuestionAnswering, 
	OwlViTProcessor, OwlViTForObjectDetection,
	MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
	CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)
vqa_model.eval()

def VQA(image: IMAGE, question: str) -> str:
	encoding = vqa_processor(image, question, return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = vqa_model.generate(**encoding)

	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
		
	return ans_op

def RESULT(var: str) -> str:
	return var

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions and variables, write a program to obtain the answer to the question. 
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. The program must assign the final answer to a variable called FINAL_ANSWER in the last line.
5. The final answer must be either "yes" or "no".
6. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images.
""",

"imple_basic_old": """You are given the following code:

from transformers import (ViltProcessor, ViltForQuestionAnswering, 
	OwlViTProcessor, OwlViTForObjectDetection,
	MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
	CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)
vqa_model.eval()

def VQA(image: IMAGE, question: str) -> str:
	encoding = vqa_processor(image, question, return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = vqa_model.generate(**encoding)

	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
		
	return ans_op

def EVAL(expr: str) -> str:
	if 'xor' in expr:
		expr = expr.replace('xor','!=')
	return str(eval(expr))

def RESULT(var: str) -> str:
	return var

The following variables are available:
RIGHT: The image on the right.
LEFT: The image on the left.

You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions, VQA(), EVAL(), and RESULT(), and variables LEFT and RIGHT, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. Only perform boolean checks by passing boolean expression to the EVAL() function. Use if-else for string comparison. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument. The input argument when calling EVAL() must be a string created using local variables and concatenation using + operators.
"""
}

def create_prompt(inputs,prompt_type='demos',num_prompts=8,method='all',seed=42,group=0):
	if prompt_type=='demos':
		if method=='random':
			random.seed(seed)
			prompt_examples = random.sample(NLVR_CURATED_EXAMPLES,num_prompts)
		elif method=='all':
			prompt_examples = NLVR_CURATED_EXAMPLES
		else:
			raise NotImplementedError
		prompt_examples = '\n'.join(prompt_examples)
		prompt_examples = f'Think step by step if the statement is True or False.\n\n{prompt_examples}'
		prompt = prompt_examples + "\nStatement: {statement}\nProgram:".format(**inputs)
	else:
		prompt = prompt_dict[prompt_type] + "\nStatement: {statement}\nProgram:\n```Python".format(**inputs)

	return prompt

# "imple_basic": """You are given the following code:

# from transformers import (ViltProcessor, ViltForQuestionAnswering, 
# 	OwlViTProcessor, OwlViTForObjectDetection,
# 	MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
# 	CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
# vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)
# vqa_model.eval()

# def VQA(image: IMAGE, question: str) -> str:
# 	encoding = vqa_processor(image, question, return_tensors='pt')
# 	encoding = {k:v.to(device) for k,v in encoding.items()}
# 	with torch.no_grad():
# 		outputs = vqa_model.generate(**encoding)

# 	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
		
# 	return ans_op

# def RESULT(var: str) -> str:
# 	return var

# The following variables are available:
# RIGHT: The image on the right.
# LEFT: The image on the left.

# Important note about VQA() function: For binary yes/no questions, the function returns "Yes" or "No".
# You are given two images 'LEFT' and 'RIGHT' and a question about them. Using only the above functions, VQA(), and RESULT(), and variables LEFT and RIGHT, write a program to obtain the answer to the question. The program must assign the final answer (True/False) to a variable called FINAL_ANSWER. If the question does not mention a specific image, you must check for both the LEFT and the RIGHT images. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
# """