import random

PROMPT = """Think step by step to carry out the instruction.

Instruction: Tag the presidents of US
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='presidents of the US',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the wild animals
Program:
OBJ0=LOC(image=IMAGE,object='wild animal')
LIST0=LIST(query='wild animals',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the shoes with their colors
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='colors',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the shoes by their type
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='type of shoes',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the shoes (4) by their type
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='type of shoes',max=4)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag oscar winning hollywood actors
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='oscar winning hollywood actors',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag these dishes with their cuisines
Program:
OBJ0=LOC(image=IMAGE,object='dish')
LIST0=LIST(query='cuisines',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the utensils used for drinking
Program:
OBJ0=LOC(image=IMAGE,object='utensil')
LIST0=LIST(query='utensils used for drinking',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the logos that have a shade of blue
Program:
OBJ0=LOC(image=IMAGE,object='logo')
LIST0=LIST(query='logos that have a shade of blue',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the logos (10) that have a shade of blue
Program:
OBJ0=LOC(image=IMAGE,object='logo')
LIST0=LIST(query='logos that have a shade of blue',max=10)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag these leaders with the countries they represent
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='countries',max={list_max})
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the actor who played Harry Potter
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='actor who played Harry Potter',max=1)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag the 7 dwarfs in Snow White
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='dwarfs in snow white',max=7)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: {instruction}
Program:
"""

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
IMAGE: The image.

You are given an image 'IMAGE' and an instruction for it. Using only the above functions and variables, write a program to carry out the instruction. The program must assign the final resulting image to a variable called FINAL_RESULT. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument. Use the FACEDET function instead of LOC when localizing faces or people.
""",

"docstring_basic": """You are given the following functions:

def LOC(image: IMAGE, object: str) -> BOX:
	# Returns the bounding boxes of the object in the image. The output of this function must be passed as an argument to one of the cropping functions or the COUNT function or both.

def FACEDET(image: IMAGE) -> OBJ:
	# Detects the faces in the image and provides bounding boxes. The output of this function must be passed as the object argument to the CLASSIFY function.

def LIST(query: str, max: int) -> List:
	# Returns a list of objects retrieved according to the query. The output of this function must be passed as the categories argument to the CLASSIFY function.

def CLASSIFY(image: IMAGE, object: OBJ, categories: List) -> OBJ:
	# Classifies the objects in the image according to the categories. Takes lists of object regions and categories and assigns one of the categories to each region. The output of this function must be passed as the object argument to the TAG function.

def TAG(image: IMAGE, object: OBJ) -> IMAGE:
	# Tags the objects in the image. Takes a list of object regions and tags each region with the corresponding category. The output of this function must be passed as the var argument to the RESULT function.

def RESULT(var: IMAGE) -> IMAGE:
	# Returns the final image.

The following variables are available:
IMAGE: The image.

You are given an image 'IMAGE' and an instruction for it. Using only the above functions and variables, write a program to carry out the instruction. The program must assign the final resulting image to a variable called FINAL_RESULT. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument. Use the FACEDET function instead of LOC when localizing faces or people.
""",

"docstring_basic_os": """You are given the following functions:

def LOC(image: IMAGE, object: str) -> BOX:
	# Returns the bounding boxes of the object in the image. The output of this function must be passed as an argument to one of the cropping functions or the COUNT function or both.

def FACEDET(image: IMAGE) -> OBJ:
	# Detects the faces in the image and provides bounding boxes. The output of this function must be passed as the object argument to the CLASSIFY function.

def LIST(query: str, max: int) -> List:
	# Returns a list of objects retrieved according to the query. The output of this function must be passed as the categories argument to the CLASSIFY function.

def CLASSIFY(image: IMAGE, object: OBJ, categories: List) -> OBJ:
	# Classifies the objects in the image according to the categories. Takes lists of object regions and categories and assigns one of the categories to each region. The output of this function must be passed as the object argument to the TAG function.

def TAG(image: IMAGE, object: OBJ) -> IMAGE:
	# Tags the objects in the image. Takes a list of object regions and tags each region with the corresponding category. The output of this function must be passed as the var argument to the RESULT function.

def RESULT(var: IMAGE) -> IMAGE:
	# Returns the final image.

The following variables are available:
IMAGE: The image.

You are given an image stored in variable 'IMAGE' and an instruction for it. By calling only the above functions and variables, write a program to carry out the instruction.
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. Use the FACEDET function instead of LOC when localizing faces or people.
5. The program must assign the final resulting image to a variable called FINAL_RESULT.
""",

"imple_basic": """You are given the following code:

import cv2
import os
import torch
import openai
import functools
import numpy as np
import face_detection
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
	OwlViTProcessor, OwlViTForObjectDetection,
	MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
	CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline
from word2number import w2n
import re

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

device = "cuda:0" if torch.cuda.is_available() else "cpu"

loc_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
loc_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
loc_model.eval()
loc_thresh = 0.1
loc_nms_thresh = 0.5

fd_model = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

classify_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
classify_model.eval()
classify_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

def LOC(img, obj_name):
	bboxes = predict_location(img,obj_name)

	objs = []
	for box in bboxes:
		objs.append(dict(
			box=box,
			category=obj_name
		))

	return objs

def enlarge_face(box,W,H,f=1.5):
	x1,y1,x2,y2 = box
	w = int((f-1)*(x2-x1)/2)
	h = int((f-1)*(y2-y1)/2)
	x1 = max(0,x1-w)
	y1 = max(0,y1-h)
	x2 = min(W,x2+w)
	y2 = min(H,y2+h)
	return [x1,y1,x2,y2]

def FACEDET(image):
	with torch.no_grad():
		faces = fd_model.detect(np.array(image))
		
		W,H = image.size
		objs = []
		for i,box in enumerate(faces):
			x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
			x1,y1,x2,y2 = enlarge_face([x1,y1,x2,y2],W,H)
			mask = np.zeros([H,W]).astype(float)
			mask[y1:y2,x1:x2] = 1.0
			objs.append(dict(
				box=[x1,y1,x2,y2],
				category='face',
				inst_id=i,
				mask = mask
			))
		return objs

def LIST(query, max):
	prompt_template = \"\"\"
Create comma separated lists based on the query.

Query: List at most 3 primary colors separated by commas
List:
red, blue, green

Query: List at most 2 north american states separated by commas
List:
California, Washington

Query: List at most {list_max} {text} separated by commas
List:\"\"\"

	response = openai.Completion.create(
		model="text-davinci-003",
		prompt=prompt_template.format(list_max=max,text=query),
		temperature=0.7,
		max_tokens=256,
		top_p=0.5,
		frequency_penalty=0,
		presence_penalty=0,
		n=1,
	)

	item_list = response.choices[0]['text'].lstrip('\n').rstrip('\n').split(', ')
	return item_list

def calculate_sim(inputs):
	img_feats = classify_model.get_image_features(inputs['pixel_values'])
	text_feats = classify_model.get_text_features(inputs['input_ids'])
	img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
	text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
	return torch.matmul(img_feats,text_feats.t())

def CLASSIFY(img, objs, query):
	if len(objs)==0:
		images = [img]
		return []
	else:
		images = [img.crop(obj['box']) for obj in objs]

	if len(query)==1:
		query = query + ['other']

	text = [f'a photo of {q}' for q in query]
	inputs = classify_processor(
		text=text, images=images, return_tensors="pt", padding=True)
	inputs = {k:v.to(device) for k,v in inputs.items()}
	with torch.no_grad():
		sim = calculate_sim(inputs)
		

	# if only one query then select the object with the highest score
	if len(query)==1:
		scores = sim.cpu().numpy()
		obj_ids = scores.argmax(0)
		obj = objs[obj_ids[0]]
		obj['class']=query[0]
		obj['class_score'] = 100.0*scores[obj_ids[0],0]
		return [obj]

	# assign the highest scoring class to each object but this may assign same class to multiple objects
	scores = sim.cpu().numpy()
	cat_ids = scores.argmax(1)
	for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
		class_name = query[cat_id]
		class_score = scores[i,cat_id]
		obj['class'] = class_name #+ f'({score_str})'
		obj['class_score'] = round(class_score*100,1)

	# sort by class scores and then for each class take the highest scoring object
	objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
	objs = [obj for obj in objs if 'class' in obj]
	classes = set([obj['class'] for obj in objs])
	new_objs = []
	for class_name in classes:
		cls_objs = [obj for obj in objs if obj['class']==class_name]

		max_score = 0
		max_obj = None
		for obj in cls_objs:
			if obj['class_score'] > max_score:
				max_obj = obj
				max_score = obj['class_score']

		new_objs.append(max_obj)

	return new_objs

def TAG(img, objs):
	W,H = img.size
	img1 = img.copy()
	draw = ImageDraw.Draw(img1)
	font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 16)
	for i,obj in enumerate(objs):
		box = obj['box']
		draw.rectangle(box,outline='green',width=4)
		x1,y1,x2,y2 = box
		label = obj['class'] + '({})'.format(obj['class_score'])
		if 'class' in obj:
			w,h = font.getsize(label)
			if x1+w > W or y2+h > H:
				draw.rectangle((x1, y2-h, x1 + w, y2), fill='green')
				draw.text((x1,y2-h),label,fill='white',font=font)
			else:
				draw.rectangle((x1, y2, x1 + w, y2 + h), fill='green')
				draw.text((x1,y2),label,fill='white',font=font)
	return img1

def RESULT(var):
	return var

The following variables are available:
IMAGE: The image.

You are given an image 'IMAGE' and an instruction for it. Using only the above functions LOC, FACEDET, LIST, CLASSIFY, TAG, and RESULT and variables, write a program to carry out the instruction. The program must assign the final resulting image to a variable called FINAL_RESULT. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument. Use the FACEDET function instead of LOC when localizing faces or people.
""",

"imple_basic_os": """You are given the following code:

import cv2
import os
import torch
import openai
import functools
import numpy as np
import face_detection
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
	OwlViTProcessor, OwlViTForObjectDetection,
	MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
	CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline
from word2number import w2n
import re

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

device = "cuda:0" if torch.cuda.is_available() else "cpu"

loc_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
loc_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
loc_model.eval()
loc_thresh = 0.1
loc_nms_thresh = 0.5

fd_model = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

classify_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
classify_model.eval()
classify_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

def LOC(img, obj_name):
	bboxes = predict_location(img,obj_name)

	objs = []
	for box in bboxes:
		objs.append(dict(
			box=box,
			category=obj_name
		))

	return objs

def enlarge_face(box,W,H,f=1.5):
	x1,y1,x2,y2 = box
	w = int((f-1)*(x2-x1)/2)
	h = int((f-1)*(y2-y1)/2)
	x1 = max(0,x1-w)
	y1 = max(0,y1-h)
	x2 = min(W,x2+w)
	y2 = min(H,y2+h)
	return [x1,y1,x2,y2]

def FACEDET(image):
	with torch.no_grad():
		faces = fd_model.detect(np.array(image))
		
		W,H = image.size
		objs = []
		for i,box in enumerate(faces):
			x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
			x1,y1,x2,y2 = enlarge_face([x1,y1,x2,y2],W,H)
			mask = np.zeros([H,W]).astype(float)
			mask[y1:y2,x1:x2] = 1.0
			objs.append(dict(
				box=[x1,y1,x2,y2],
				category='face',
				inst_id=i,
				mask = mask
			))
		return objs

def LIST(query, max):
	prompt_template = \"\"\"
Create comma separated lists based on the query.

Query: List at most 3 primary colors separated by commas
List:
red, blue, green

Query: List at most 2 north american states separated by commas
List:
California, Washington

Query: List at most {list_max} {text} separated by commas
List:\"\"\"

	response = openai.Completion.create(
		model="text-davinci-003",
		prompt=prompt_template.format(list_max=max,text=query),
		temperature=0.7,
		max_tokens=256,
		top_p=0.5,
		frequency_penalty=0,
		presence_penalty=0,
		n=1,
	)

	item_list = response.choices[0]['text'].lstrip('\n').rstrip('\n').split(', ')
	return item_list

def calculate_sim(inputs):
	img_feats = classify_model.get_image_features(inputs['pixel_values'])
	text_feats = classify_model.get_text_features(inputs['input_ids'])
	img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
	text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
	return torch.matmul(img_feats,text_feats.t())

def CLASSIFY(img, objs, query):
	if len(objs)==0:
		images = [img]
		return []
	else:
		images = [img.crop(obj['box']) for obj in objs]

	if len(query)==1:
		query = query + ['other']

	text = [f'a photo of {q}' for q in query]
	inputs = classify_processor(
		text=text, images=images, return_tensors="pt", padding=True)
	inputs = {k:v.to(device) for k,v in inputs.items()}
	with torch.no_grad():
		sim = calculate_sim(inputs)
		

	# if only one query then select the object with the highest score
	if len(query)==1:
		scores = sim.cpu().numpy()
		obj_ids = scores.argmax(0)
		obj = objs[obj_ids[0]]
		obj['class']=query[0]
		obj['class_score'] = 100.0*scores[obj_ids[0],0]
		return [obj]

	# assign the highest scoring class to each object but this may assign same class to multiple objects
	scores = sim.cpu().numpy()
	cat_ids = scores.argmax(1)
	for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
		class_name = query[cat_id]
		class_score = scores[i,cat_id]
		obj['class'] = class_name #+ f'({score_str})'
		obj['class_score'] = round(class_score*100,1)

	# sort by class scores and then for each class take the highest scoring object
	objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
	objs = [obj for obj in objs if 'class' in obj]
	classes = set([obj['class'] for obj in objs])
	new_objs = []
	for class_name in classes:
		cls_objs = [obj for obj in objs if obj['class']==class_name]

		max_score = 0
		max_obj = None
		for obj in cls_objs:
			if obj['class_score'] > max_score:
				max_obj = obj
				max_score = obj['class_score']

		new_objs.append(max_obj)

	return new_objs

def TAG(img, objs):
	W,H = img.size
	img1 = img.copy()
	draw = ImageDraw.Draw(img1)
	font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 16)
	for i,obj in enumerate(objs):
		box = obj['box']
		draw.rectangle(box,outline='green',width=4)
		x1,y1,x2,y2 = box
		label = obj['class'] + '({})'.format(obj['class_score'])
		if 'class' in obj:
			w,h = font.getsize(label)
			if x1+w > W or y2+h > H:
				draw.rectangle((x1, y2-h, x1 + w, y2), fill='green')
				draw.text((x1,y2-h),label,fill='white',font=font)
			else:
				draw.rectangle((x1, y2, x1 + w, y2 + h), fill='green')
				draw.text((x1,y2),label,fill='white',font=font)
	return img1

def RESULT(var):
	return var

The following variables are available:
IMAGE: The image.

You are given an image stored in variable 'IMAGE' and an instruction for it. By calling only the above functions and variables, write a program to carry out the instruction.
Follow these instructions:
1. Only generate python code.
2. Do not introduce or define new functions or import libraries.
3. Call one of the above functions in each line of the program and assign the return value to a temporary variable which will be passed as an argument to the next function call.
4. Use the FACEDET function instead of LOC when localizing faces or people.
5. The program must assign the final resulting image to a variable called FINAL_RESULT.
"""
}

def create_prompt(inputs,prompt_type='demos',num_prompts=8,method='all',seed=42,group=0):
	if prompt_type=='demos':
		if method=='all':
			prompt = PROMPT.format(**inputs)
		else:
			raise NotImplementedError
	else:
		prompt = prompt_dict[prompt_type] + "\nInstruction: {instruction}\nProgram:\n```Python".format(**inputs)

	return prompt