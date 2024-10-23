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
import pdb

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)
vqa_model.eval()

loc_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
loc_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
loc_model.eval()
loc_thresh = 0.1
loc_nms_thresh = 0.5

# fd_model = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

classify_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
classify_model.eval()
classify_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
mask_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco").to(device)
mask_model.eval()

select_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
select_model.eval()
select_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def dummy(images, **kwargs):
	return images, False

replace_pipe = StableDiffusionInpaintPipeline.from_pretrained(
	"stabilityai/stable-diffusion-2-inpainting",
	revision="fp16",
	torch_dtype=torch.float16)
replace_pipe = replace_pipe.to(device)
replace_pipe.safety_checker = dummy

def word_to_num(match):
	return str(w2n.word_to_num(match.group()))

def convert_numbers(s):
	# Replace words that represent numbers with the numbers themselves
	return re.sub(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion)+\b', word_to_num, s, flags=re.IGNORECASE)

def replace_yes_no(s):
	s = re.sub(r'\byes\b', 'True', s)
	s = re.sub(r'\bYes\b', 'True', s)
	s = re.sub(r'\bno\b', 'False', s)
	s = re.sub(r'\bNo\b', 'False', s)
	return s

def EVAL(expr):
	# print(expr)
	expr = replace_yes_no(expr)
	expr = convert_numbers(expr)
	if 'xor' in expr:
		expr = expr.replace('xor','!=')
	# print(expr)
	return str(eval(expr))

def EVAL2(expr): # for GQA
	# print(expr)
	expr = convert_numbers(expr)
	if 'xor' in expr:
		expr = expr.replace('xor','!=')
	# print(expr)
	return str(eval(expr))

def VQA(image, question):
	encoding = vqa_processor(image, question, return_tensors='pt')
	encoding = {k:v.to(device) for k,v in encoding.items()}
	with torch.no_grad():
		outputs = vqa_model.generate(**encoding)

	ans_op = vqa_processor.decode(outputs[0], skip_special_tokens=True)
	# ans_op = replace_yes_no(ans_op)
	ans_op = convert_numbers(ans_op)
		
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

def LOC(image, object):
	if object=='TOP':
		bboxes = [top_box(image)]
	elif object=='BOTTOM':
		bboxes = [bottom_box(image)]
	elif object=='LEFT':
		bboxes = [left_box(image)]
	elif object=='RIGHT':
		bboxes = [right_box(image)]
	else:
		bboxes = predict_location(image, object)

	box_img = box_image(image, bboxes)

	return bboxes

def LOC2(image, object):
	bboxes = predict_location(image,object)

	objs = []
	for box in bboxes:
		objs.append(dict(
			box=box,
			category=object
		))

	return objs

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
		box1 = box[0]
		box1 = expand_box(box1, image.size)
		out_img = image.crop(box1)
	else:
		box1 = []
		out_img = image

	return out_img

def right_of(box,img_size):
	w,h = img_size
	x1,y1,x2,y2 = box
	cx = int((x1+x2)/2)
	return [cx,0,w-1,h-1]

def CROP_RIGHTOF(image, box):
	if len(box) > 0:
		box1 = box[0]
		right_box = right_of(box1, image.size)
	else:
		w,h = image.size
		box1 = []
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
		box1 = box[0]
		left_box = left_of(box1, image.size)
	else:
		w,h = image.size
		box1 = []
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
		box1 = box[0]
		above_box = above(box1, image.size)
	else:
		w,h = img.size
		box1 = []
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
		box1 = box[0]
		below_box = below(box1, image.size)
	else:
		w,h = img.size
		box1 = []
		below_box = [0,0,int(w/2),h-1]
	
	out_img = image.crop(below_box)

	return out_img

def enlarge_face(box,W,H,f=1.5):
	x1,y1,x2,y2 = box
	w = int((f-1)*(x2-x1)/2)
	h = int((f-1)*(y2-y1)/2)
	x1 = max(0,x1-w)
	y1 = max(0,y1-h)
	x2 = min(W,x2+w)
	y2 = min(H,y2+h)
	return [x1,y1,x2,y2]

# def FACEDET(image):
# 	with torch.no_grad():
# 		faces = fd_model.detect(np.array(image))
		
# 		W,H = image.size
# 		objs = []
# 		for i,box in enumerate(faces):
# 			x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
# 			x1,y1,x2,y2 = enlarge_face([x1,y1,x2,y2],W,H)
# 			mask = np.zeros([H,W]).astype(float)
# 			mask[y1:y2,x1:x2] = 1.0
# 			objs.append(dict(
# 				box=[x1,y1,x2,y2],
# 				category='face',
# 				inst_id=i,
# 				mask = mask
# 			))
# 		return objs

def LIST(query, max):
	prompt_template = """
Create comma separated lists based on the query.

Query: List at most 3 primary colors separated by commas
List:
red, blue, green

Query: List at most 2 north american states separated by commas
List:
California, Washington

Query: List at most {list_max} {text} separated by commas
List:"""

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

def CLASSIFY(image, object, categories):
	img = image
	objs = object
	query = categories
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

def TAG(image, object):
	img = image
	objs = object
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

def TAG_EVAL(image, object):
	img = image
	objs = object
	W,H = img.size
	img1 = img.copy()
	draw = ImageDraw.Draw(img1)
	font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 16)
	tags = {}
	for i,obj in enumerate(objs):
		box = obj['box']
		x1,y1,x2,y2 = box
		label = obj['class']
		tags[label] = [x1,y1,x2,y2]
	return tags

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

def execute_program(prog, init_state):
	exec(prog, globals(), init_state)

	if "FINAL_ANSWER" in prog:
		final_answer = init_state['FINAL_ANSWER']
	elif "FINAL_RESULT" in prog:
		final_answer = init_state['FINAL_RESULT']
	else:
		var_name = prog.split("\n")[-1].split('=')[0].strip()
		final_answer = init_state[var_name]
	
	return final_answer