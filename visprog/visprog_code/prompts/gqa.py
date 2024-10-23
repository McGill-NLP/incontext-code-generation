import random

GQA_CURATED_EXAMPLES=[
"""Question: Is the vehicle in the top of the image?
Program:
BOX0=LOC(image=IMAGE,object="TOP")
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="vehicle")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Are there trains or fences in this scene?
Program:
BOX0=LOC(image=IMAGE,object="train")
BOX1=LOC(image=IMAGE,object="fence")
ANSWER0=COUNT(box=BOX0)
ANSWER1=COUNT(box=BOX1)
ANSWER2=EVAL(expr="'yes' if '{ANSWER0}' + '{ANSWER1}' > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER)
""",
"""Question: Who is carrying the umbrella?
Program:
BOX0=LOC(image=IMAGE,object="umbrella")
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question="Who is carrying the umbrella?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Which place is it?
Program:
ANSWER0=VQA(image=IMAGE,question="Which place is it?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: What color is the curtain that is to the right of the mirror?
Program:
BOX0=LOC(image=IMAGE,object="mirror")
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question="What color is the curtain?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Is the pillow in the top part or in the bottom of the picture?
Program:
BOX0=LOC(image=IMAGE,object="TOP")
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="pillow")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'top' if '{ANSWER0}' > 0 else 'bottom'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Question: Do you see bottles to the right of the wine on the left of the picture?
Program:
BOX0=LOC(image=IMAGE,object="LEFT")
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="wine")
IMAGE1=CROP_RIGHTOF(image=IMAGE0,box=BOX1)
BOX2=LOC(image=IMAGE1,object="bottles")
ANSWER0=COUNT(box=BOX2)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is the street light standing behind a truck?
Program:
BOX0=LOC(image=IMAGE,object="truck")
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="street light")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Which side is the food on?
Program:
BOX0=LOC(image=IMAGE,object="RIGHT")
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="food")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'right' if '{ANSWER0}' > 0 else 'left'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What do the wetsuit and the sky have in common?
Program:
ANSWER0=VQA(image=IMAGE,question="What do the wetsuit and the sky have in common?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Do the post and the sign have a different colors?
Program:
BOX0=LOC(image=IMAGE,object="post")
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object="sign")
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question="What color is the post?")
ANSWER1=VQA(image=IMAGE1,question="What color is the sign?")
ANSWER2=EVAL(expr="'yes' if '{ANSWER0}' != '{ANSWER1}' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Does the traffic cone have white color?
Program:
BOX0=LOC(image=IMAGE,object="traffic cone")
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question="What color is the traffic cone?")
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' == 'white' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Are these animals of different species?
Program:
ANSWER0=VQA(image=IMAGE,question="Are these animals of different species?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Which side of the image is the chair on?
Program:
BOX0=LOC(image=IMAGE,object="RIGHT")
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="chair")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'right' if '{ANSWER0}' > 0 else 'left'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Do you see any drawers to the left of the plate?
Program:
BOX0=LOC(image=IMAGE,object="plate")
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="drawers")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Does the mat have the same color as the sky?
Program:
BOX0=LOC(image=IMAGE,object="sky")
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object="mat")
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question="What color is the sky?")
ANSWER1=VQA(image=IMAGE1,question="What color is the mat?")
ANSWER2=EVAL(expr="'yes' if '{ANSWER0}' == '{ANSWER1}' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Is a cat above the mat?
Program:
BOX0=LOC(image=IMAGE,object="mat")
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="cat")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
"""
"""Question: Is the cat above a mat?
Program:
BOX0=LOC(image=IMAGE,object="cat")
IMAGE0=CROP_BELOW(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="mat")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 and else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is the mat below a cat?
Program:
BOX0=LOC(image=IMAGE,object="mat")
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="cat")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is a mat below the cat?
Program:
BOX0=LOC(image=IMAGE,object="cat")
IMAGE0=CROP_BELOW(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object="mat")
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' > 0 and else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
]

prompt_dict = {
"zero-shot-code": """You are given an image denoted by the variable 'IMAGE' and a question about it. Write a program to obtain the answer to the question. You can import and use any function or library you want. The program must assign the final answer to a variable called FINAL_ANSWER.
""",

"zero-shot": """You are given a question. To the best of your ability, provide an answer to the question. Take a guess if you don't know. Limit your answer to 1-2 words.
""",

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
IMAGE: The image.

You are given an image 'IMAGE' and a question about it. Using only the above functions and variables, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
""",

"docstring_basic": """You are given the following functions:

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

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
IMAGE: The image.

You are given an image 'IMAGE' and a question about it. Using only the above functions and variables, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
""",

"docstring_basic_os": """You are given the following functions:

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

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
IMAGE: The image.

You are given an image stored in variable 'IMAGE' and a question about it. By calling only the above functions and variables, write a program to obtain the answer to the question.
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. The program must assign the final answer to a variable called FINAL_ANSWER in the last line.
""",

"docstring_basic_os_10ex": """You are given the following functions:

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

def RESULT(var: str) -> str:
	# Returns the final answer.

Some examples of programs using the above functions and variables:
example 1:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > <number> else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 2:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 3:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object=<specified object>)
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=VQA(image=IMAGE1,question=<subquestion>)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)

example 4:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 5:
ANSWER0=VQA(image=IMAGE,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 6:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 7:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' == <specified color> else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 8:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 9:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 10:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
IMAGE1=CROP_RIGHTOF(image=IMAGE0,box=BOX1)
BOX2=LOC(image=IMAGE1,object=<specified object>)
ANSWER0=COUNT(box=BOX2)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

The following variables are available:
IMAGE: The image.

You are given an image stored in variable 'IMAGE' and a question about it. By calling only the above functions and variables, write a program to obtain the answer to the question.
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. The program must assign the final answer to a variable called FINAL_ANSWER in the last line.
""",

"docstring_indi_1ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.
Example:
ANSWER2=VQA(image=IMAGE1,question=<subquestion>)

def LOC(image: IMAGE, object: str) -> BOX:
	# Returns the bounding box of the object in the image.
Example:
BOX1=LOC(image=IMAGE,object=<specified object>)

def COUNT(box: BOX) -> int:
	# Returns the number of objects detected.
Example:
ANSWER0=COUNT(box=BOX1)

def CROP(image: IMAGE, box: BOX) -> IMAGE:
	# Returns the cropped image corresponding to the bounding boxes.
Example:
IMAGE1=CROP(image=IMAGE,box=BOX1)

def CROP_LEFTOF(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is on the left of the bounding box.
Example:
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)

def CROP_RIGHTOF(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is on the right of the bounding box.
Example:
IMAGE2=CROP_RIGHTOF(image=IMAGE1,box=BOX1)

def CROP_ABOVE(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is above the bounding box.
Example:
IMAGE1=CROP_ABOVE(image=IMAGE0,box=BOX1)

def CROP_BELOW(image: IMAGE, box: BOX) -> IMAGE:
	# Crops the image to keep the portion that is below the bounding box.
Example:
IMAGE0=CROP_BELOW(image=IMAGE0,box=BOX0)

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.
Example:
ANSWER5=EVAL(expr='{ANSWER1} <boolean operator> {ANSWER3}')

def RESULT(var: str) -> str:
	# Returns the final answer.
Example:
FINAL_ANSWER=RESULT(var=ANSWER6)

The following variables are available:
IMAGE: The image.

You are given an image 'IMAGE' and a question about it. Using only the above functions and variable, write a program that returns the answer to the question. Perform boolean checks by passing input to the EVAL() function.
""",

"docstring_1ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def LOC(image: IMAGE, object: str) -> BOX:
	# Returns the bounding box of the object in the image.

def COUNT(box: BOX) -> int:
	# Returns the number of objects detected.

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

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
IMAGE: The image.

One example of a program using the above functions and variables:

BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object=<specified object>)
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=VQA(image=IMAGE1,question=<subquestion>)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)

You are given an image 'IMAGE' and a question about it. Using only the above functions and variable, write a program that returns the answer to the question. Perform boolean checks by passing input to the EVAL() function.
""",

"docstring_3ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def LOC(image: IMAGE, object: str) -> BOX:
	# Returns the bounding box of the object in the image.

def COUNT(box: BOX) -> int:
	# Returns the number of objects detected.

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

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
IMAGE: The image.

Some examples of programs using the above functions and variables:
example 1:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > <number> else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 2:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 3:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object=<specified object>)
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=VQA(image=IMAGE1,question=<subquestion>)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)

You are given an image 'IMAGE' and a question about it. Using only the above functions and variable, write a program that returns the answer to the question. Perform boolean checks by passing input to the EVAL() function.
""",

"docstring_10ex": """You are given the following functions:

def VQA(image: IMAGE, question: str) -> str:
	# Returns the answer to the question about the image.

def LOC(image: IMAGE, object: str) -> BOX:
	# Returns the bounding box of the object in the image.

def COUNT(box: BOX) -> int:
	# Returns the number of objects detected.

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

def EVAL(expr: str) -> str:
	# The input expression is parsed as a python expression and evaluated using the python eval() function. The expression can include normal operators in python such as or, and, etc as well as the exclusive-or for xor operations.

def RESULT(var: str) -> str:
	# Returns the final answer.

The following variables are available:
IMAGE: The image.

Some examples of programs using the above functions and variables:
example 1:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > <number> else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 2:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 3:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object=<specified object>)
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=VQA(image=IMAGE1,question=<subquestion>)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)

example 4:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 5:
ANSWER0=VQA(image=IMAGE,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 6:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 7:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' == <specified color> else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 8:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 9:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 10:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
IMAGE1=CROP_RIGHTOF(image=IMAGE0,box=BOX1)
BOX2=LOC(image=IMAGE1,object=<specified object>)
ANSWER0=COUNT(box=BOX2)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

You are given an image 'IMAGE' and a question about it. Using only the above functions and variable, write a program that returns the answer to the question. Perform boolean checks by passing input to the EVAL() function.
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

loc_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
loc_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
loc_model.eval()
loc_thresh = 0.1
loc_nms_thresh = 0.5

def VQA(image: IMAGE, question: str) -> str:
	encoding = vqa_processor(image, question, return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = vqa_model.generate(**encoding)

	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
		
	return ans_op

def top_box(img):
	w,h = img.size
	return [0,0,w-1,int(h/2)]

def bottom_box(img):
	w,h = img.size
	return [0,int(h/2),w-1,h-1]

def left_box(img):
	w,h = img.size
	return [0,0,int(w/2),h-1]

def right_box(img):
	w,h = img.size
	return [int(w/2),0,w-1,h-1]

def box_image(img,boxes,highlight_best=True):
	img1 = img.copy()
	draw = ImageDraw.Draw(img1)
	for i,box in enumerate(boxes):
		if i==0 and highlight_best:
			color = 'red'
		else:
			color = 'blue'

		draw.rectangle(box,outline=color,width=5)

	return img1

def normalize_coord(bbox,img_size):
	w,h = img_size
	x1,y1,x2,y2 = [int(v) for v in bbox]
	x1 = max(0,x1)
	y1 = max(0,y1)
	x2 = min(x2,w-1)
	y2 = min(y2,h-1)
	return [x1,y1,x2,y2]

def predict_location(img, obj_name):
	encoding = loc_processor(
		text=[[f'a photo of {obj_name}']], 
		images=img,
		return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = loc_model(**encoding)
		for k,v in outputs.items():
			if v is not None:
				outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
	
	target_sizes = torch.Tensor([img.size[::-1]])
	results = loc_processor.post_process_object_detection(outputs=outputs,threshold=loc_thresh,target_sizes=target_sizes)
	boxes, scores = results[0]["boxes"], results[0]["scores"]
	boxes = boxes.cpu().detach().numpy().tolist()
	scores = scores.cpu().detach().numpy().tolist()
	if len(boxes)==0:
		return []

	boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
	selected_boxes = []
	selected_scores = []
	for i in range(len(scores)):
		if scores[i] > loc_thresh:
			coord = normalize_coord(boxes[i],img.size)
			selected_boxes.append(coord)
			selected_scores.append(scores[i])

	selected_boxes, selected_scores = nms(selected_boxes,selected_scores,loc_nms_thresh)
	return selected_boxes

def LOC(image, obj_name):
	if obj_name=='TOP':
		bboxes = [top_box(image)]
	elif obj_name=='BOTTOM':
		bboxes = [bottom_box(image)]
	elif obj_name=='LEFT':
		bboxes = [left_box(image)]
	elif obj_name=='RIGHT':
		bboxes = [right_box(image)]
	else:
		bboxes = predict_location(image, obj_name)

	box_img = box_image(image, bboxes)

	return bboxes

def COUNT(box):
	count = len(box)
	return count

def expand_box(box,img_size,factor=1.5):
	W,H = img_size
	x1,y1,x2,y2 = box
	dw = int(factor*(x2-x1)/2)
	dh = int(factor*(y2-y1)/2)
	cx = int((x1 + x2) / 2)
	cy = int((y1 + y2) / 2)
	x1 = max(0,cx - dw)
	x2 = min(cx + dw,W)
	y1 = max(0,cy - dh)
	y2 = min(cy + dh,H)
	return [x1,y1,x2,y2]

def CROP(image, box):
	if len(box) > 0:
		box = box[0]
		box = expand_box(box, image.size)
		out_img = image.crop(box)
	else:
		box = []
		out_img = image

	return out_img

def right_of(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cx = int((x1+x2)/2)
	return [cx,0,w-1,h-1]

def CROP_RIGHTOF(image, box):
	if len(box) > 0:
		box = box[0]
		right_box = right_of(box, image.size)
	else:
		w,h = image.size
		box = []
		right_box = [int(w/2),0,w-1,h-1]
	
	out_img = image.crop(right_box)

	return out_img

def left_of(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cx = int((x1+x2)/2)
	return [0,0,cx,h-1]

def CROP_LEFTOF(image, box):
	if len(box) > 0:
		box = box[0]
		left_box = left_of(box, image.size)
	else:
		w,h = image.size
		box = []
		left_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(left_box)

	return out_img

def above(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cy = int((y1+y2)/2)
	return [0,0,w-1,cy]

def CROP_ABOVE(image, box):
	if len(box) > 0:
		box = box[0]
		above_box = above(box, image.size)
	else:
		w,h = img.size
		box = []
		above_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(above_box)

	return out_img

def below(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cy = int((y1+y2)/2)
	return [0,cy,w-1,h-1]

def CROP_BELOW(image, box):
	if len(box) > 0:
		box = box[0]
		below_box = below(box, image.size)
	else:
		w,h = img.size
		box = []
		below_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(below_box)

	return out_img

def RESULT(var: str) -> str:
	return var

The following variables are available:
IMAGE: The image.

You are given an image 'IMAGE' and a question about it. Using only the above functions and variables, write a program to obtain the answer to the question. The program must assign the final answer to a variable called FINAL_ANSWER. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
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

loc_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
loc_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
loc_model.eval()
loc_thresh = 0.1
loc_nms_thresh = 0.5

def VQA(image: IMAGE, question: str) -> str:
	encoding = vqa_processor(image, question, return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = vqa_model.generate(**encoding)

	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
		
	return ans_op

def top_box(img):
	w,h = img.size
	return [0,0,w-1,int(h/2)]

def bottom_box(img):
	w,h = img.size
	return [0,int(h/2),w-1,h-1]

def left_box(img):
	w,h = img.size
	return [0,0,int(w/2),h-1]

def right_box(img):
	w,h = img.size
	return [int(w/2),0,w-1,h-1]

def box_image(img,boxes,highlight_best=True):
	img1 = img.copy()
	draw = ImageDraw.Draw(img1)
	for i,box in enumerate(boxes):
		if i==0 and highlight_best:
			color = 'red'
		else:
			color = 'blue'

		draw.rectangle(box,outline=color,width=5)

	return img1

def normalize_coord(bbox,img_size):
	w,h = img_size
	x1,y1,x2,y2 = [int(v) for v in bbox]
	x1 = max(0,x1)
	y1 = max(0,y1)
	x2 = min(x2,w-1)
	y2 = min(y2,h-1)
	return [x1,y1,x2,y2]

def predict_location(img, obj_name):
	encoding = loc_processor(
		text=[[f'a photo of {obj_name}']], 
		images=img,
		return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = loc_model(**encoding)
		for k,v in outputs.items():
			if v is not None:
				outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
	
	target_sizes = torch.Tensor([img.size[::-1]])
	results = loc_processor.post_process_object_detection(outputs=outputs,threshold=loc_thresh,target_sizes=target_sizes)
	boxes, scores = results[0]["boxes"], results[0]["scores"]
	boxes = boxes.cpu().detach().numpy().tolist()
	scores = scores.cpu().detach().numpy().tolist()
	if len(boxes)==0:
		return []

	boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
	selected_boxes = []
	selected_scores = []
	for i in range(len(scores)):
		if scores[i] > loc_thresh:
			coord = normalize_coord(boxes[i],img.size)
			selected_boxes.append(coord)
			selected_scores.append(scores[i])

	selected_boxes, selected_scores = nms(selected_boxes,selected_scores,loc_nms_thresh)
	return selected_boxes

def LOC(image, obj_name):
	if obj_name=='TOP':
		bboxes = [top_box(image)]
	elif obj_name=='BOTTOM':
		bboxes = [bottom_box(image)]
	elif obj_name=='LEFT':
		bboxes = [left_box(image)]
	elif obj_name=='RIGHT':
		bboxes = [right_box(image)]
	else:
		bboxes = predict_location(image, obj_name)

	box_img = box_image(image, bboxes)

	return bboxes

def COUNT(box):
	count = len(box)
	return count

def expand_box(box,img_size,factor=1.5):
	W,H = img_size
	x1,y1,x2,y2 = box
	dw = int(factor*(x2-x1)/2)
	dh = int(factor*(y2-y1)/2)
	cx = int((x1 + x2) / 2)
	cy = int((y1 + y2) / 2)
	x1 = max(0,cx - dw)
	x2 = min(cx + dw,W)
	y1 = max(0,cy - dh)
	y2 = min(cy + dh,H)
	return [x1,y1,x2,y2]

def CROP(image, box):
	if len(box) > 0:
		box = box[0]
		box = expand_box(box, image.size)
		out_img = image.crop(box)
	else:
		box = []
		out_img = image

	return out_img

def right_of(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cx = int((x1+x2)/2)
	return [cx,0,w-1,h-1]

def CROP_RIGHTOF(image, box):
	if len(box) > 0:
		box = box[0]
		right_box = right_of(box, image.size)
	else:
		w,h = image.size
		box = []
		right_box = [int(w/2),0,w-1,h-1]
	
	out_img = image.crop(right_box)

	return out_img

def left_of(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cx = int((x1+x2)/2)
	return [0,0,cx,h-1]

def CROP_LEFTOF(image, box):
	if len(box) > 0:
		box = box[0]
		left_box = left_of(box, image.size)
	else:
		w,h = image.size
		box = []
		left_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(left_box)

	return out_img

def above(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cy = int((y1+y2)/2)
	return [0,0,w-1,cy]

def CROP_ABOVE(image, box):
	if len(box) > 0:
		box = box[0]
		above_box = above(box, image.size)
	else:
		w,h = img.size
		box = []
		above_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(above_box)

	return out_img

def below(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cy = int((y1+y2)/2)
	return [0,cy,w-1,h-1]

def CROP_BELOW(image, box):
	if len(box) > 0:
		box = box[0]
		below_box = below(box, image.size)
	else:
		w,h = img.size
		box = []
		below_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(below_box)

	return out_img

def RESULT(var: str) -> str:
	return var

The following variables are available:
IMAGE: The image.

You are given an image stored in variable 'IMAGE' and a question about it. By calling only the above functions and variables, write a program to obtain the answer to the question.
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. The program must assign the final answer to a variable called FINAL_ANSWER in the last line.
""",

"imple_basic_os_10ex": """You are given the following code:

from transformers import (ViltProcessor, ViltForQuestionAnswering, 
	OwlViTProcessor, OwlViTForObjectDetection,
	MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
	CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)
vqa_model.eval()

loc_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
loc_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
loc_model.eval()
loc_thresh = 0.1
loc_nms_thresh = 0.5

def VQA(image: IMAGE, question: str) -> str:
	encoding = vqa_processor(image, question, return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = vqa_model.generate(**encoding)

	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
		
	return ans_op

def top_box(img):
	w,h = img.size
	return [0,0,w-1,int(h/2)]

def bottom_box(img):
	w,h = img.size
	return [0,int(h/2),w-1,h-1]

def left_box(img):
	w,h = img.size
	return [0,0,int(w/2),h-1]

def right_box(img):
	w,h = img.size
	return [int(w/2),0,w-1,h-1]

def box_image(img,boxes,highlight_best=True):
	img1 = img.copy()
	draw = ImageDraw.Draw(img1)
	for i,box in enumerate(boxes):
		if i==0 and highlight_best:
			color = 'red'
		else:
			color = 'blue'

		draw.rectangle(box,outline=color,width=5)

	return img1

def normalize_coord(bbox,img_size):
	w,h = img_size
	x1,y1,x2,y2 = [int(v) for v in bbox]
	x1 = max(0,x1)
	y1 = max(0,y1)
	x2 = min(x2,w-1)
	y2 = min(y2,h-1)
	return [x1,y1,x2,y2]

def predict_location(img, obj_name):
	encoding = loc_processor(
		text=[[f'a photo of {obj_name}']], 
		images=img,
		return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = loc_model(**encoding)
		for k,v in outputs.items():
			if v is not None:
				outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
	
	target_sizes = torch.Tensor([img.size[::-1]])
	results = loc_processor.post_process_object_detection(outputs=outputs,threshold=loc_thresh,target_sizes=target_sizes)
	boxes, scores = results[0]["boxes"], results[0]["scores"]
	boxes = boxes.cpu().detach().numpy().tolist()
	scores = scores.cpu().detach().numpy().tolist()
	if len(boxes)==0:
		return []

	boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
	selected_boxes = []
	selected_scores = []
	for i in range(len(scores)):
		if scores[i] > loc_thresh:
			coord = normalize_coord(boxes[i],img.size)
			selected_boxes.append(coord)
			selected_scores.append(scores[i])

	selected_boxes, selected_scores = nms(selected_boxes,selected_scores,loc_nms_thresh)
	return selected_boxes

def LOC(image, obj_name):
	if obj_name=='TOP':
		bboxes = [top_box(image)]
	elif obj_name=='BOTTOM':
		bboxes = [bottom_box(image)]
	elif obj_name=='LEFT':
		bboxes = [left_box(image)]
	elif obj_name=='RIGHT':
		bboxes = [right_box(image)]
	else:
		bboxes = predict_location(image, obj_name)

	box_img = box_image(image, bboxes)

	return bboxes

def COUNT(box):
	count = len(box)
	return count

def expand_box(box,img_size,factor=1.5):
	W,H = img_size
	x1,y1,x2,y2 = box
	dw = int(factor*(x2-x1)/2)
	dh = int(factor*(y2-y1)/2)
	cx = int((x1 + x2) / 2)
	cy = int((y1 + y2) / 2)
	x1 = max(0,cx - dw)
	x2 = min(cx + dw,W)
	y1 = max(0,cy - dh)
	y2 = min(cy + dh,H)
	return [x1,y1,x2,y2]

def CROP(image, box):
	if len(box) > 0:
		box = box[0]
		box = expand_box(box, image.size)
		out_img = image.crop(box)
	else:
		box = []
		out_img = image

	return out_img

def right_of(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cx = int((x1+x2)/2)
	return [cx,0,w-1,h-1]

def CROP_RIGHTOF(image, box):
	if len(box) > 0:
		box = box[0]
		right_box = right_of(box, image.size)
	else:
		w,h = image.size
		box = []
		right_box = [int(w/2),0,w-1,h-1]
	
	out_img = image.crop(right_box)

	return out_img

def left_of(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cx = int((x1+x2)/2)
	return [0,0,cx,h-1]

def CROP_LEFTOF(image, box):
	if len(box) > 0:
		box = box[0]
		left_box = left_of(box, image.size)
	else:
		w,h = image.size
		box = []
		left_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(left_box)

	return out_img

def above(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cy = int((y1+y2)/2)
	return [0,0,w-1,cy]

def CROP_ABOVE(image, box):
	if len(box) > 0:
		box = box[0]
		above_box = above(box, image.size)
	else:
		w,h = img.size
		box = []
		above_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(above_box)

	return out_img

def below(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cy = int((y1+y2)/2)
	return [0,cy,w-1,h-1]

def CROP_BELOW(image, box):
	if len(box) > 0:
		box = box[0]
		below_box = below(box, image.size)
	else:
		w,h = img.size
		box = []
		below_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(below_box)

	return out_img

def RESULT(var: str) -> str:
	return var

Some examples of programs using the above functions and variables:
example 1:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > <number> else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 2:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 3:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object=<specified object>)
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=VQA(image=IMAGE1,question=<subquestion>)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)

example 4:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 5:
ANSWER0=VQA(image=IMAGE,question=<subquestion>)
FINAL_RESULT=RESULT(var=ANSWER0)

example 6:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 7:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question=<subquestion>)
ANSWER1=EVAL(expr="'yes' if '{ANSWER0}' == <specified color> else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 8:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 9:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

example 10:
BOX0=LOC(image=IMAGE,object=<specified object>)
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object=<specified object>)
IMAGE1=CROP_RIGHTOF(image=IMAGE0,box=BOX1)
BOX2=LOC(image=IMAGE1,object=<specified object>)
ANSWER0=COUNT(box=BOX2)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

The following variables are available:
IMAGE: The image.

You are given an image stored in variable 'IMAGE' and a question about it. By calling only the above functions and variables, write a program to obtain the answer to the question.
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. The program must assign the final answer to a variable called FINAL_ANSWER in the last line.
"""
}

def create_prompt(inputs,prompt_type='demos',num_prompts=8,method='all',seed=42,group=0):
	if prompt_type=='demos':
		if method=='all':
			prompt_examples = GQA_CURATED_EXAMPLES
		elif method=='random':
			random.seed(seed)
			prompt_examples = random.sample(GQA_CURATED_EXAMPLES,num_prompts)
		else:
			raise NotImplementedError
		prompt_examples = '\n'.join(prompt_examples)
		prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'
		prompt = prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)
	else:
		prompt = prompt_dict[prompt_type] + "\nQuestion: {question}\nProgram:\n```Python".format(**inputs)

	return prompt