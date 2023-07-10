import numpy as np
import os
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import draw_bounding_boxes
import string
import objectDetector
import util.utils as util
import PIL
import random
import time
from skimage.transform import resize
import sys

import cv2

class TemporalGraph:
	def __init__(self, device, buffer_size, OBJECTS_ALLOWED, N, STRIDE, model='yolov5', BBOX_FEATURES=True): #model = 'fasterrcnn_resnet50_fpn_coco'):
		self.features = {}
		self.DEVICE = device
		self.OBJECTS_ALLOWED = np.asarray(OBJECTS_ALLOWED)
		self.model_type = model
		object_detector = objectDetector.ObjectDetector(device, model=self.model_type)
		self.model = object_detector.getModel().to(self.DEVICE)

		self.OBJECT_DETECTION_THESHOLD = 0.55
		#self.path_training_normal = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames/training/normal"		
		#self.path_training_abnormal = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames/training/anomaly"				
		#self.path_test_normal = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames/test/normal"				
		#self.path_test_abnormal = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames/test/anomaly"				
		self.N = N 			# For each frame, we will consider the top N scores objects
		self.STRIDE = STRIDE
		self.buffer_size = buffer_size
		self.buffer = []	# [[folder_index, img_index], [boxes1, scores1, labels1, bbox_fea_vec1]]
							#				ID             , 				CNN result
		self.BBOX_FEATURES = BBOX_FEATURES

	# Get only the top 'self.N' objects with better scores
	def filterLowScores(self, prediction):

		# read coco labels
		str_labels = np.asarray(util.fileLines2List("../files/coco_labels.txt"))

		scores = prediction[0]['scores'].cpu().detach().numpy()
		boxes = prediction[0]['boxes'].cpu().detach().numpy()
		labels = prediction[0]['labels'].cpu().detach().numpy()
		bbox_fea_vec = prediction[0]['bbox_fea_vec'].cpu().detach().numpy()     # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP
		#str_labels1 = str_labels[labels1-1]


		# Let's remove objects not interesting
		inds = [i in self.OBJECTS_ALLOWED for i in labels]
		print(bbox_fea_vec)
		scores, boxes, labels, bbox_fea_vec = scores[inds], boxes[inds], labels[inds], bbox_fea_vec[inds]
		

		# Here we have the object list ordered by scores
		new_boxes = []
		new_scores = []
		new_labels = []
		new_bbox_fea_vec = []
		
		# Every object have to be considered in the adjacency graph
		# The top-N object have to be get after
		new_boxes = boxes[0:]#self.N]
		new_scores = scores[0:]#self.N]
		new_labels = labels[0:]#self.N]
		new_bbox_fea_vec = bbox_fea_vec[0:]#self.N]

		"""
		new_boxes = []
		new_scores = []
		new_labels = []
		new_bbox_fea_vec = []
		for i in range(boxes.shape[0]):
			if scores[i] > self.OBJECT_DETECTION_THESHOLD:
				new_boxes.append(boxes[i])
				new_scores.append(scores[i])
				new_labels.append(labels[i])
				new_bbox_fea_vec.append(bbox_fea_vec[i])
		"""


		return np.asarray(new_boxes), np.asarray(new_scores), np.asarray(new_labels), np.asarray(new_bbox_fea_vec)


	def countFiles(self, path, extension):
		counter = 0
		for img in os.listdir(path):
			ex = img.lower().rpartition('.')[-1]

			if ex == extension:
				counter += 1
		return counter


	def extractFrames(self, path, img_counter):

		# Read the 'img_counter' image
		name = os.path.join(path, str(img_counter) + '.png')
		img1 = cv2.imread(name)
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 

		# Read the next image
		name = os.path.join(path, str(img_counter+1) + '.png')
		img2 = cv2.imread(name)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 		



		return (img1, img2)

	def inference(self, image):
		if self.model_type == 'fasterrcnn_resnet50_fpn_coco':
			return self.inference_resnet(image)
		elif self.model_type == 'yolov5':
			return self.inference_yolo(image)
		print("No model detected")
		exit()


	# Inferene
	def inference_resnet(self, image):	
		print("inference by ResNet")
		original_image = image

		# convert the image from BGR to RGB channel ordering and change the
		# image from channels last to channels first ordering
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#image = image.view(2, 0, 1)

		image = torch.permute(image, (2, 0, 1))
		#image= torch.transpose(image, 0, 2)


		# add the batch dimension, scale the raw pixel intensities to the
		# range [0, 1], and convert the image to a floating point tensor
		image = torch.unsqueeze(image, axis=0)

		image = image / 255.0
		

		#image = torch.FloatTensor(image)

		# send the input to the device and pass the it through the network to
		# get the detections and predictions
		image = image.to(self.DEVICE)

		prediction = self.model(image)    


		"""
		APAGAR DEPOID
		"""
		#scores = prediction[0]['scores'].cpu().detach().numpy()
		#boxes = prediction[0]['boxes'].cpu().detach().numpy()
		#labels = prediction[0]['labels'].cpu().detach().numpy()
		#print(labels)

		# read coco labels
		#str_labels = np.asarray(fileLines2List("coco_labels.txt"))
		#str_labels = str_labels[labels-1]


		# TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP
		#bbox_fea_vec = prediction[0]['bbox_fea_vec'].cpu().detach().numpy()


		#boxes, scores, labels, _ = self.filterLowScores(prediction)
		#boxes = torch.from_numpy(boxes) 
		#scores = torch.from_numpy(scores)  
		#labels = labels 

		#original_image = original_image.cpu().numpy().astype('uint8')
		
		#image_tensor = torch.from_numpy(original_image)
		#image_tensor = torch.moveaxis(image_tensor, 2, 0)
		
		#print(image_tensor.shape)
		#print(image_tensor.shape)
		#labels = list(map(str, labels))
		#print(labels)
		#img_com_bb = draw_bounding_boxes(image_tensor, boxes, labels, font="Plane Crash.ttf")
		#img_com_bb = torch.moveaxis(img_com_bb, 0, 2)


		#img_com_bb = img_com_bb.numpy()

		#rd = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))

		#PIL.Image.fromarray(img_com_bb).convert("RGB").save("imagens/art"+rd+".png")


		return prediction

	
	def get_features(self, name):
		def hook(model, input, output):
			self.features[name] = output
		return hook

	def yolov5_result(self, images, fea=False):

		#res = model(images)
		sys.path.append("/media/denis/dados/CamNuvem")
		from yolov5.utils.general import (Profile, cv2, non_max_suppression)		
		seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

		with dt[0]:
			images = images.half() if self.model.fp16 else images.float()

		# Inference
		with dt[1]:
			save_dir = '.'
			path = '.'

			#image = Image.open("pessoa.jpg")
			#image = np.array(image)
			#image = resize(image, (HEIGHT, WIDTH))
			#image = np.transpose(image, (2, 0, 1))
			#image = torch.from_numpy(image[None]).float()
			#images = image
			#print(images.shape)

			self.model.model.model[23].register_forward_hook(self.get_features('fea'))
			pred = self.model(images, augment=False, visualize=False)
			features = self.features['fea'][0]
			
			avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
			features = avgpool(features)	# (1280, 1, 1)

			# Feature maps das imagens
	
			#exit()
			#print(self.features['Detect'][2][0].shape)
			#print(self.features['Detect'][2][1].shape)
			#print(self.features['Detect'][2][2].shape)

		if fea == True:
			return features

		agnostic_nms = False
		conf_thres=0.25  # confidence threshold
		iou_thres=0.45  # NMS IOU threshold
		max_det=1000  # maximum detections per image
		classes=None
		# NMS
		with dt[2]:	
			pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


		return pred
				

	def inference_yolo(self, image):	
		original_image = image

		# convert the image from BGR to RGB channel ordering and change the
		# image from channels last to channels first ordering
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#image = image.view(2, 0, 1)

		image = torch.permute(image, (2, 0, 1))
		original_image = torch.permute(original_image, (2, 0, 1))
		#image= torch.transpose(image, 0, 2)

		# add the batch dimension, scale the raw pixel intensities to the
		# range [0, 1], and convert the image to a floating point tensor
		image = torch.unsqueeze(image, axis=0)
		original_image = torch.unsqueeze(original_image, axis=0)

		image = image / 255.0
		
		#image = torch.FloatTensor(image)

		print(image.shape)
		new_image_list = []
		if True:
			for image_ in image:
				image_ = image_.cpu().numpy()
			
				original_size_h = image_.shape[1]
				original_size_w = image_.shape[2]
				if image_.shape[1] != 640 or image_.shape[0] != 640:
					image_ = np.transpose(image_, (1, 2, 0))
					new_image = resize(image_, (640, 640))		# Receives (H, W, C)
					new_image = np.transpose(new_image, (2, 0, 1))

				new_image_list.append(new_image)
			
			image = np.stack(new_image_list, axis=0)
			image = torch.from_numpy(image).float()
		

		# send the input to the device and pass the it through the network to
		# get the detections and predictions
		images = image.to(self.DEVICE)

		WIDTH = 640
		HEIGHT = 640

		print("Shape entrando no YOLO")

		pred = self.yolov5_result(images)	# images = [1,3,640,640]

		"""
		Put in the resnet prediction format:
		scores = prediction[0]['scores'].cpu().detach().numpy()
		boxes = prediction[0]['boxes'].cpu().detach().numpy()
		labels = prediction[0]['labels'].cpu().detach().numpy()
		bbox_fea_vec = prediction[0]['bbox_fea_vec'].cpu().detach().numpy()     # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP
		"""
		scores = []
		boxes = []
		labels = []
		bbox_fea_vec = []

		prediction = []
		cont = 0
		for *xyxy, conf, cls in reversed(pred[0]):
			xyxy = [float(i) for i in xyxy]
			
			# For some reason, some coordinates can be negative. If this happens, we replace it to 0
			xyxy_ = []
			for i in xyxy:
				if i < 0:
					xyxy_.append(0)
				else:
					xyxy_.append(i)
			xyxy = xyxy_

			xyxy_resized = []

			print(xyxy)
			# make the xyxy coords in the range [0, 320], [0,240]
			xyxy_resized.append((xyxy[0]/WIDTH)*original_size_w)
			xyxy_resized.append((xyxy[1]/HEIGHT)*original_size_h)
			xyxy_resized.append((xyxy[2]/WIDTH)*original_size_w)
			xyxy_resized.append((xyxy[3]/HEIGHT)*original_size_h)

			print(xyxy_resized)

			conf = float(conf)
			cls = float(cls)
			boxes.append(xyxy_resized)
			scores.append(conf)
			labels.append(int(cls))

			from PIL import Image as im

			# Initially, lets disregard the bbox features 
			if self.BBOX_FEATURES == True:
				print("Vamos usar as features dos bbox")
				object_crop_list = []
#				for bbox in xyxy:
				# cut the image on this bbox

				image_bbox = image[0, :, int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
				oi = image_bbox.numpy()
				
				
				oi = np.transpose(oi, (1,2,0))
				oi = resize(oi, (640, 640))

				#oi = (oi*255).astype(np.uint8)
				
				#data = im.fromarray(oi)
				#data.save("teste"+str(cont)+".png")

				oi = torch.from_numpy(np.transpose(oi, (2, 0, 1)))
				oi = oi[None, :, :, :].to(self.DEVICE)

				pred = self.yolov5_result(oi, fea=True)
				pred = pred[:,0,0]

				print(type(pred))
				print(pred.shape)

				bbox_fea_vec.append(pred.cpu().numpy())
				

				cont += 1

				# resize to WIDTHxHEIGHT

				# YoloV5 feature vector extraction
			else:
				print("Não vamos usar as features dos bbox")
				bbox_fea_vec.append(np.ones(1280))


		# YOLO give labels 0-based. We consider labels 1 based. So we have +1 on labels
		aux = {'scores': torch.Tensor(scores), 'boxes': torch.Tensor(boxes), 'labels': torch.Tensor(labels).type(torch.uint8)+1, 'bbox_fea_vec': torch.Tensor(bbox_fea_vec)}
		prediction.append(aux)
		#prediction[0]['scores'] = scores
		#prediction[0]['boxes'] = boxes
		#prediction[0]['labels'] = labels
		#prediction[0]['bbox_fea_vec'] = bbox_fea_vec
		
		return prediction



		## APAGAR DEPOIS!!!!
		"""
		scores = prediction[0]['scores'].cpu().detach().numpy()
		boxes = prediction[0]['boxes'].cpu().detach().numpy()
		labels = prediction[0]['labels'].cpu().detach().numpy()
		print(labels)

		# read coco labels
		str_labels = np.asarray(utils.fileLines2List("../files/coco_labels.txt"))
		str_labels = str_labels[labels]


		# TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP
		bbox_fea_vec = prediction[0]['bbox_fea_vec'].cpu().detach().numpy()


		boxes, scores, labels, _ = self.filterLowScores(prediction)
		boxes = torch.from_numpy(boxes) 
		scores = torch.from_numpy(scores)  
		labels = labels 
		# 		
		original_image = original_image.cpu().numpy().astype('uint8')
		
		image_tensor = torch.from_numpy(original_image)
		image_tensor = torch.moveaxis(image_tensor, 2, 0)
		
		print(image_tensor.shape)
		print(image_tensor.shape)
		labels = list(map(str, labels))


		img_com_bb = draw_bounding_boxes((image.squeeze()*255).cpu().type(torch.uint8), boxes, labels, font="Plane Crash.ttf")
		img_com_bb = torch.moveaxis(img_com_bb, 0, 2)


		img_com_bb = img_com_bb.numpy()

		rd = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))

		PIL.Image.fromarray(img_com_bb).convert("RGB").save("imagens/art"+rd+".png")


		return prediction
		"""


		

	# (boxes1, scores1, labels1, bbox_fea_vec1)
	# Given the all object feature vector from two consecutives frame, get the similarity
	# between all of then and apply a threshold to re-ientify a object at these frames.
	def make_temporal_graph(self, pred1, pred2):

		# read coco labels
		str_labels = np.asarray(util.fileLines2List("../files/coco_labels.txt"))

		scores1 = pred1[1]          # pred1[0]['scores'].cpu().detach().numpy()
		boxes1 = pred1[0]           # pred1[0]['boxes'].cpu().detach().numpy()
		labels1 = pred1[2]          # pred1[0]['labels'].cpu().detach().numpy()
		bbox_fea_vec1 = pred1[3]     #pred1[0]['bbox_fea_vec'].cpu().detach().numpy()     # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP

		scores2 = pred2[1]          # pred2[0]['scores'].cpu().detach().numpy()
		boxes2 = pred2[0]           # pred2[0]['boxes'].cpu().detach().numpy()
		labels2 = pred2[2]          # pred2[0]['labels'].cpu().detach().numpy()     
		bbox_fea_vec2 = pred2[3]    # pred2[0]['bbox_fea_vec'].cpu().detach().numpy()     # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP

		# If some image has no object, the graph is empty
		if len(labels1) == 0 or len(labels2) == 0:
			return []


		####
		# Compute appearence distance between each object
		####	
		str_labels1 = str_labels[labels1-1]
		str_labels2 = str_labels[labels2-1]

		cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
		mat = np.zeros((bbox_fea_vec1.shape[0], bbox_fea_vec2.shape[0]))

		#          vec2[0] vec2[1] vec2[2] ..
		#vec1[0]     x       y        z   
		#vec1[1]    ...
		#vec1[2]
		# ...
		bbox_fea_vec1 = torch.from_numpy(bbox_fea_vec1)
		bbox_fea_vec2 = torch.from_numpy(bbox_fea_vec2)
		vec1_ = torch.repeat_interleave(bbox_fea_vec1, bbox_fea_vec2.shape[0], dim=0)

		vec2_ = bbox_fea_vec2.repeat(bbox_fea_vec1.shape[0], 1)

		appea_dist = cos(vec1_, vec2_)

		# Yeeep, here we have a adjacency matrix
		appea_dist = appea_dist.view(bbox_fea_vec1.shape[0], bbox_fea_vec2.shape[0])

		####
		# Compute spacial distance similarity between each object
		####
		frame1_boxes = []
		for box in boxes1:
			dx = box[2] - box[0]
			dy = box[3] - box[1]
			x_center = box[0] + dx/2
			y_center = box[1] + dy/2

			frame1_boxes.append([x_center, y_center, dx, dy])

		frame2_boxes = []
		for box in boxes2:
			dx = box[2] - box[0]
			dy = box[3] - box[1]
			x_center = box[0] + dx/2
			y_center = box[1] + dy/2

			#print(x_center)
			#print(y_center)
			#print(dx)
			#print(dy)
			#print("\n\n")

			frame2_boxes.append([x_center, y_center, dx, dy])

		frame1_boxes = torch.from_numpy(np.asarray(frame1_boxes))
		frame2_boxes = torch.from_numpy(np.asarray(frame2_boxes))

		vec1_ = torch.repeat_interleave(frame1_boxes, frame2_boxes.shape[0], dim=0)
		vec2_ = frame2_boxes.repeat(frame1_boxes.shape[0], 1)

		spacial_sim = cos(vec1_, vec2_)
		#spacial_sim = torch.dist(vec1_, vec2_, p=2)
		#spacial_sim = torch.cdist(vec1_, vec2_, p=2.0)
		spacial_sim = spacial_sim.view(bbox_fea_vec1.shape[0], bbox_fea_vec2.shape[0])

		"""
		print(spacial_sim.shape)
		print("Primeiro valor do vec_1")
		print(vec1_[0])
		print("Primeiro valor do vec_2")
		print(vec2_[3])		
		print("Cos entre vec1 e vec2")
		print(spacial_sim[0][3])
		print("Similaridade entre")
		print(appea_dist[0][3])

		print("Score entre o 0 e 0")
		print(0.4*spacial_sim[0][0] + 0.6*appea_dist[0][0])
		print("Score entre o 0 e 0 3")
		print(0.4*spacial_sim[0][3] + 0.6*appea_dist[0][3])

		exit()	
		"""

		# The spacial_sim presents a very low alteration between two different points, so we have to use
		# a exponential function to valorize this small difference

		if self.model_type == "fasterrcnn_resnet50_fpn_coco":
			output = (0.4 * (50**(spacial_sim)-49) + 0.6*appea_dist)	#/2
		else:
			#output = spacial_sim #(0.4 * spacial_sim + 0.6*appea_dist)	#/2
			output = (0.4 * (50**(spacial_sim)-49) + 0.6*appea_dist)	#/2

		return output

	def acessBuffer(self, key, img):
		elements = [i[0] for i in self.buffer]	

		# If this sample is in buffer, we do not need run inference again, lets only get the result from buffer
		if key in elements:
			index = elements.index(key)
			boxes1, scores1, labels1, bbox_fea_vec1 = self.buffer[index][1]
		else:
			prediction = self.inference(img)
			boxes1, scores1, labels1, bbox_fea_vec1  = self.filterLowScores(prediction)      
				
			# If this image does not exist in buffer, add it
			if len(self.buffer) == self.buffer_size:
				del self.buffer[0]

			self.buffer.append([key, [boxes1, scores1, labels1, bbox_fea_vec1]])		

		return boxes1, scores1, labels1, bbox_fea_vec1

	# receive a set of frames [T, W, H, C]
	# return a set of T-1 adjacency matrix connecting every object in a frame pair
	def frames2temporalGraph(self, images, folder_index, sample_index):
		ma = []
		num_img = images.shape[0]
		print("num oimage")
		print(num_img)
		bbox_fea_list = []
		box_list = []
		score_list = []

		for i in range(num_img-1):
			img1, img2 = images[i], images[i+1]

			print(img1.shape)
			print(img2.shape)

			# Verify if img1 exists in self.buffer
			img_index = sample_index + i

			key = [folder_index, img_index]
			
			boxes1, scores1, labels1, bbox_fea_vec1 = self.acessBuffer(key, img1)
			data1 = (boxes1, scores1, labels1, bbox_fea_vec1) 

			# Verify if img1 exists in self.buffer
			img_index = sample_index + i+1

			key = [folder_index, img_index]

			boxes2, scores2, labels2, bbox_fea_vec2 = self.acessBuffer(key, img2)	
			data2 = (boxes2, scores2, labels2, bbox_fea_vec2) 		

			bbox_fea_list.append([bbox_fea_vec1, bbox_fea_vec2])
			box_list.append([boxes1, boxes2])
			score_list.append([scores1, scores2])
			
			adjacency_matrix = self.make_temporal_graph(data1, data2)

			if len(bbox_fea_vec2) == 0 : 
				if len(adjacency_matrix) > 0:
					print('ERROR')
					exit()

			ma.append(adjacency_matrix)
			#path_ad_mat = os.path.join(foldername, str(i)+'.adj.npy')			

		return ma, bbox_fea_list, box_list, score_list
