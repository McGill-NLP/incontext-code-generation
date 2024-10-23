PROMPT = """Think step by step to carry out the instruction.

Emoji Options: 
:p = face_with_tongue
8) = smiling_face_with_sunglasses
:) = smiling_face
;) = winking_face

Instruction: Hide the face of Nicole Kidman with :p
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Nicole Kidman',category=None)
IMAGE0=EMOJI(image=IMAGE,object=OBJ1,emoji='face_with_tongue')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Hide the faces of Nicole Kidman and Brad Pitt with ;) and 8)
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Nicole Kidman',category=None)
IMAGE0=EMOJI(image=IMAGE,object=OBJ1,emoji='winking_face')
OBJ2=SELECT(image=IMAGE,object=OBJ0,query='Brad Pitt',category=None)
IMAGE1=EMOJI(image=IMAGE0,object=OBJ1,emoji='smiling_face_with_sunglasses')
FINAL_RESULT=RESULT(var=IMAGE1)

Instruction: Create a color pop of Amy and Daphne
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Amy,Daphne',category=None)
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Create a color pop of the girl and the umbrella
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='girl,umbrella',category=None)
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Create a color pop of the dog, frisbee, and grass
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='dog,frisbee,grass',category=None)
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Create a color pop of the man wearing a red suit (person)
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='man wearing a red suit',category='person')
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Select the red bus and blur the background
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=BGBLUR(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Replace the red bus with a blue bus
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Replace the red bus with blue bus and the road with dirt road
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
OBJ2=SEG(image=IMAGE0)
OBJ3=SELECT(image=IMAGE0,object=OBJ2,query='road',category=None)
IMAGE1=REPLACE(image=IMAGE0,object=OBJ3,prompt='dirt road')
FINAL_RESULT=RESULT(var=IMAGE1)

Instruction: Replace the red bus (bus) with a truck
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category='bus')
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
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

Replace the emojis with their corresponding names in the program: 
:p = face_with_tongue
8) = smiling_face_with_sunglasses
:) = smiling_face
;) = winking_face

You are given an image 'IMAGE' and an instruction for it. Using only the above functions and variables, write a program to carry out the instruction. The program must assign the final resulting image to a variable called FINAL_RESULT. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
""",

"docstring_basic": """You are given the following functions:

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
	
def RESULT(var: IMAGE) -> IMAGE:
	# Returns the final image.

The following variables are available:
IMAGE: The image.

Replace the emojis with their corresponding names in the program: 
:p = face_with_tongue
8) = smiling_face_with_sunglasses
:) = smiling_face
;) = winking_face

You are given an image 'IMAGE' and an instruction for it. Using only the above functions and variables, write a program to carry out the instruction. The program must assign the final resulting image to a variable called FINAL_RESULT. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
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

fd_model = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
mask_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco").to(device)
mask_model.eval()

select_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
select_model.eval()
select_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def dummy(images, **kwargs):
	return images, False

replace_pipe = StableDiffusionInpaintPipeline.from_pretrained(
	"runwayml/stable-diffusion-inpainting",
	revision="fp16",
	torch_dtype=torch.float16)
replace_pipe = replace_pipe.to(device)
replace_pipe.safety_checker = dummy

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

def pred_seg(img):
		inputs = feature_extractor(images=img, return_tensors="pt")
		inputs = {k:v.to(device) for k,v in inputs.items()}
		with torch.no_grad():
			outputs = mask_model(**inputs)
		outputs = feature_extractor.post_process_panoptic_segmentation(outputs)[0]
		instance_map = outputs['segmentation'].cpu().numpy()
		objs = []
		print(outputs.keys())
		for seg in outputs['segments_info']:
			inst_id = seg['id']
			label_id = seg['label_id']
			category = mask_model.config.id2label[label_id]
			mask = (instance_map==inst_id).astype(float)
			resized_mask = np.array(
				Image.fromarray(mask).resize(
					img.size,resample=Image.BILINEAR))
			Y,X = np.where(resized_mask>0.5)
			x1,x2 = np.min(X), np.max(X)
			y1,y2 = np.min(Y), np.max(Y)
			num_pixels = np.sum(mask)
			objs.append(dict(
				mask=resized_mask,
				category=category,
				box=[x1,y1,x2,y2],
				inst_id=inst_id
			))

		return objs

def SEG(image):
	inputs = feature_extractor(images=image, return_tensors="pt")
	inputs = {k:v.to(device) for k,v in inputs.items()}
	with torch.no_grad():
		outputs = mask_model(**inputs)
	outputs = feature_extractor.post_process_panoptic_segmentation(outputs)[0]
	instance_map = outputs['segmentation'].cpu().numpy()
	objs = []
	print(outputs.keys())
	for seg in outputs['segments_info']:
		inst_id = seg['id']
		label_id = seg['label_id']
		category = mask_model.config.id2label[label_id]
		mask = (instance_map==inst_id).astype(float)
		resized_mask = np.array(
			Image.fromarray(mask).resize(
				image.size,resample=Image.BILINEAR))
		Y,X = np.where(resized_mask>0.5)
		x1,x2 = np.min(X), np.max(X)
		y1,y2 = np.min(Y), np.max(Y)
		num_pixels = np.sum(mask)
		objs.append(dict(
			mask=resized_mask,
			category=category,
			box=[x1,y1,x2,y2],
			inst_id=inst_id
		))

	return objs

def query_string_match(objs,q):
	obj_cats = [obj['category'] for obj in objs]
	q = q.lower()
	for cat in [q,f'{q}-merged',f'{q}-other-merged']:
		if cat in obj_cats:
			return [obj for obj in objs if obj['category']==cat]
	
	return None

def select_calculate_sim(inputs):
	img_feats = select_model.get_image_features(inputs['pixel_values'])
	text_feats = select_model.get_text_features(inputs['input_ids'])
	img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
	text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
	return torch.matmul(img_feats,text_feats.t())

def query_obj(query,objs,img):
	images = [img.crop(obj['box']) for obj in objs]
	text = [f'a photo of {q}' for q in query]
	inputs = select_processor(
		text=text, images=images, return_tensors="pt", padding=True)
	inputs = {k:v.to(device) for k,v in inputs.items()}
	with torch.no_grad():
		scores = select_calculate_sim(inputs).cpu().numpy()
		
	obj_ids = scores.argmax(0)
	return [objs[i] for i in obj_ids]

def SELECT(image, object, query, category):
	img = image
	objs = object
	query = query.split(',')
	select_objs = []

	if category is not None:
		cat_objs = [obj for obj in objs if obj['category'] in category]
		if len(cat_objs) > 0:
			objs = cat_objs

	if category is None:
		for q in query:
			matches = query_string_match(objs, q)
			if matches is None:
				continue
			select_objs += matches

	if query is not None and len(select_objs)==0:
		select_objs = query_obj(query, objs, img)

	return select_objs


def EMOJI(image, object, emoji):
	img = image
	objs = object
	emoji_name = emoji
	W,H = img.size
	emojipth = os.path.join(EMOJI_DIR,f'smileys/{emoji_name}.png')
	for obj in objs:
		x1,y1,x2,y2 = obj['box']
		cx = (x1+x2)/2
		cy = (y1+y2)/2
		s = (y2-y1)/1.5
		x_pos = (cx-0.5*s)/W
		y_pos = (cy-0.5*s)/H
		emoji_size = s/H
		emoji_aug = imaugs.OverlayEmoji(
			emoji_path=emojipth,
			emoji_size=emoji_size,
			x_pos=x_pos,
			y_pos=y_pos)
		img = emoji_aug(img)

	return img


def refine_mask(img,mask):
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	mask,_,_ = cv2.grabCut(
		img.astype(np.uint8),
		mask.astype(np.uint8),
		None,
		bgdModel,
		fgdModel,
		5,
		cv2.GC_INIT_WITH_MASK)
	return mask.astype(float)

def COLORPOP(image, object):
	gimg = image.copy()
	gimg = gimg.convert('L').convert('RGB')
	gimg = np.array(gimg).astype(float)
	img = np.array(image).astype(float)
	for obj in object:
		refined_mask = refine_mask(img, obj['mask'])
		mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
		gimg = mask*img + (1-mask)*gimg

	gimg = np.array(gimg).astype(np.uint8)
	gimg = Image.fromarray(gimg)

	return gimg


def smoothen_mask(mask):
	mask = Image.fromarray(255*mask.astype(np.uint8)).filter(
		ImageFilter.GaussianBlur(radius = 5))
	return np.array(mask).astype(float)/255

def BGBLUR(image, object):
	bgimg = image.copy()
	bgimg = bgimg.filter(ImageFilter.GaussianBlur(radius = 2))
	bgimg = np.array(bgimg).astype(float)
	img = np.array(image).astype(float)
	for obj in object:
		refined_mask = refine_mask(img, obj['mask'])
		mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
		mask = smoothen_mask(mask)
		bgimg = mask*img + (1-mask)*bgimg

	bgimg = np.array(bgimg).astype(np.uint8)
	bgimg = Image.fromarray(bgimg)

	return bgimg


def create_mask_img(objs):
	mask = objs[0]['mask']
	mask[mask>0.5] = 255
	mask[mask<=0.5] = 0
	mask = mask.astype(np.uint8)
	return Image.fromarray(mask)

def resize_and_pad(img,size=(512,512)):
	new_img = Image.new(img.mode,size)
	thumbnail = img.copy()
	thumbnail.thumbnail(size)
	new_img.paste(thumbnail,(0,0))
	W,H = thumbnail.size
	return new_img, W, H

def replace_predict(img,mask,prompt):
	mask,_,_ = resize_and_pad(mask)
	init_img,W,H = resize_and_pad(img)
	new_img = replace_pipe(
		prompt=prompt,
		image=init_img,
		mask_image=mask,
		# strength=0.98,
		guidance_scale=7.5,
		num_inference_steps=50 #200
	).images[0]
	return new_img.crop((0,0,W-1,H-1)).resize(img.size)

def REPLACE(image, object, prompt):
	mask = create_mask_img(object)
	new_img = replace_predict(image, mask, prompt)
	return new_img

def RESULT(var):
	return var

The following variables are available:
IMAGE: The image.

Replace the emojis with their corresponding names in the program: 
:p = face_with_tongue
8) = smiling_face_with_sunglasses
:) = smiling_face
;) = winking_face

You are given an image 'IMAGE' and an instruction for it. Using only the above functions FACEDET, SEG, SELECT, EMOJI, COLORPOP, BGBLUR, REPLACE, RESULT and variables, write a program to carry out the instruction. The program must assign the final resulting image to a variable called FINAL_RESULT. The value obtained by calling any of the above functions must first be stored in a temporary variable before being passed as an argument.
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