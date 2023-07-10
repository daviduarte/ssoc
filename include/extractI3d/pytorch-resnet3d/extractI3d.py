import imageio
from models import resnet
from models import C3D_model
#from models import tsm as TSM
#from models.tsm import init_model as TSM
import cv2
from PIL import Image
import numpy as np
import math
import torch
import magic
import time
from moviepy.editor import *
from torchvision import transforms
import sys
from feature_extractors.tsm import init_model as TSM
current_dir = os.getcwd()
print(os.path.join(current_dir, 'anomalyDetection/graph_detector'))
sys.path.append(os.path.join(current_dir, 'anomalyDetection/graph_detector'))
import temporalGraph


ROOT_DIR = os.path.abspath(os.curdir)
sys.path.append(os.path.join(ROOT_DIR, "../../"))

#from inferencia import inferencia	# Interface para usar o RTFM
import gtransformers


NUM_FRAMES_IN_EACH_FEATURE_VECTOR = 16
FRAMES_SIZE = 256
TEN_CROP_SIZE = 224


#i3d_path_test = "/home/denis/Documentos/CamNuvem/violent_thefts_dataset/i3d/test"
#i3d_path_training = "/home/denis/Documentos/CamNuvem/violent_thefts_dataset/i3d/test"



"""
def extractFeatureVector(b_data):
	# b_data is tensor of dimension 1*3*16*224*224
	b_data = torch.from_numpy(b_data)
	b_data = Variable(b_data.cuda(), volatile=True).float()
	inp = {'frames': b_data}
	features = net(inp)
	# features is tensor of dimension 1*2048*1*1*1
	return features
"""	

def readVideo(video):

	cap = cv2.VideoCapture(video)
	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

	fc = 0
	ret = True

	while (fc < frameCount  and ret):
	    ret, buf[fc] = cap.read()
	    fc += 1

	cap.release()

	#cv2.namedWindow('frame 10')
	#cv2.imshow('frame 10', buf[9])
	return buf

"""
def resizeVideo(videoArray):
	newVideoArray = []
	for i in range(videoArray.shape[0]):
		resized = cv2.resize(videoArray[i], (224, 224), interpolation=cv2.INTER_CUBIC)

		newVideoArray.append(resized)
		
	return newVideoArray
"""
# videoArray: (frames, 10, C, W, H) or (frames, C, W, H)
# return: list of arrays: [(32, 10, C, W, H), (), ...]. Note the 32 in the begin of array. This is a design prior from Sultani's (and others) Paper.
# The last segment may has less than 32 frames
def divideVideo(videoArray):
	print("DIVIDE VIDEOS ***")
	print(videoArray.shape)

	numFrames = videoArray.shape[0]
	count = 0
	countGlobal = 0
	lista = []

	input_ = []

	for i in range(numFrames):

		lista.append(videoArray[i])

		count += 1	

		if count == NUM_FRAMES_IN_EACH_FEATURE_VECTOR:
			lista = np.stack(lista, 0)
			input_.append(lista)
			count = 0
			lista = []

		# The last segment may has less frames than NUM_FRAMES_IN_EACH_FEATURE_VECTOR. So we sample the last frame until reach the desirable NUM_FRAMES_IN_EACH_FEATURE_VECTOR
		if i == numFrames-1 and count > 0:
			for j in range((NUM_FRAMES_IN_EACH_FEATURE_VECTOR - count)):
				lista.append(lista[len(lista)-1])
			lista = np.stack(lista, 0)
			input_.append(lista)



	print("Lista 32 croppado")
	print(input_[0].shape)

	return input_

def init_tsm_model(gpu_id):
	net_tsm, crop_size_tsm, scale_size_tsm, input_mean_tsm, input_std_tsm = TSM(num_class=2048, num_segments=16, 
	                                                        modality='RGB', 
	                                                        arch='resnet50',  # 'resnet101'
	                                                        consensus_type='avg', # 'avg' 'identity'
	                                                        dropout=0, 
	                                                        img_feature_dim=256,
	                                                        no_partialbn=True, #False,
	                                                        pretrain='imagenet',
	                                                        is_shift=True, 
	                                                        shift_div=8, 
	                                                        shift_place='blockers', 
	                                                        non_local=False,
	                                                        temporal_pool=False,
	                                                        resume_chkpt='/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/extractI3d/pytorch-resnet3d/pretrained/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth',
	                                                        #get_global_pool=True,
	                                                        gpus=[gpu_id])	

	return net_tsm, crop_size_tsm, scale_size_tsm, input_mean_tsm, input_std_tsm


def chooseModel(feature_extractor, gpu_id):
	if feature_extractor == 'tsm' or feature_extractor == 'tsm-2048':
		return init_tsm_model(gpu_id)

	elif feature_extractor == 'c3d':
		print("Choosing c3d")
		return C3D_model.C3D(487) # C3D

	elif feature_extractor == 'i3d':
		return resnet.i3_res50(400) # vanilla I3D ResNet50

	elif feature_extractor == "sshc":
		input = []

		buffer_size = 1		# Just used for buffer. Not used here
		DEVICE = "cuda:0"
		OBJECTS_ALLOWED = [1,2,3,4]    # COCO categories ID allowed. The othwers will be discarded
		N = 5
		STRIDE = -1			# Not used here

		temporal_graph = temporalGraph.TemporalGraph(DEVICE, buffer_size, OBJECTS_ALLOWED, N, STRIDE)
		return temporal_graph
		

# img tem  (B, 3, 224, 224)
def start(imgs, ten_crop, feature_extractor, gpu_id, model):

	print("Entrou no VIDEO_PATH")

	#video = readVideo(video_path)

	#video = resizeVideo(video)
	#print(video)

	# Transforma lista de numpy array em um único np array
	#video = np.stack(video, 0)

	#inputs = divideVideo(video)
	inputs = divideVideo(imgs)

	print("Shape antes da rede")
	print(inputs[0].shape) # (t, b, c, w, h) (32, 10, 3, 224, 224)

	test_counter = 0

	feature_vector = []


	for input_ in inputs:
		# input_ (t, b, c, w, h) (32, 10, 3, 224, 224)
		
		if not ten_crop:
			input_ = input_[:,None,:,:,:]

		input_ = np.moveaxis(input_, [0,1,2,3,4], [2, 0, 1, 3, 4])	# b,c,t,h,w  # 10x3x32x224x224

		input_ = input_.astype('float32')
		print(input_.shape)

		input_ = torch.from_numpy(input_)

		device = torch.device(gpu_id)
		if feature_extractor == 'i3d':

			net_i3d = model
			net_i3d.to(device)
			input_ = input_.to(device)
			net_i3d.train(False)  # Set model to evaluate mode


			input_ = {'frames': input_}

			output = net_i3d(input_)
			output = output[0].detach().cpu().numpy()
			output = output[:, :, 0, 0, 0]

		elif feature_extractor == 'c3d':

			net_c3d = model
			net_c3d = net_c3d.to(device)
			net_c3d.eval()
			net_c3d.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'extractI3d/pytorch-resnet3d/pretrained/c3d.pickle')))			

			EXTRACTED_LAYER = 5

			print("Tamanho do input: ")
			print(input_.element_size() * input_.nelement())
			print("Tamanh do modelo")
			print(sys.getsizeof(net_c3d))
			if input_.size()[0] == 10:
				print("kkkk")
				new_input = input_[0:2, :, :, :, :]
				output = net_c3d(new_input, EXTRACTED_LAYER)

				new_input = input_[2:4, :, :, :, :]
				output = net_c3d(new_input, EXTRACTED_LAYER)

				new_input = input_[4:6, :, :, :, :]
				output = net_c3d(new_input, EXTRACTED_LAYER)

				new_input = input_[6:8, :, :, :, :]
				output = net_c3d(new_input, EXTRACTED_LAYER)

				new_input = input_[8:10, :, :, :, :]
				output = net_c3d(new_input, EXTRACTED_LAYER)												
			else:
				input_ = input_.to(device)
				print("input kkk: ")
				print(input_.shape)
				output = net_c3d(input_, EXTRACTED_LAYER)
				print(output[1].shape)
				exit()
				

			print(output[0])
			output = output[0].detach().cpu().numpy() 
			print(output.shape)
			output = output[0,0,:,:]

		elif feature_extractor == 'tsm' or feature_extractor == 'tsm-2048':
			
			net_tsm, crop_size_tsm, scale_size_tsm, input_mean_tsm, input_std_tsm = model

			print("chegou aki")
			print()
			#clip_frames = torch.from_numpy(np.ones((16,3,256,256)).astype('float32'))
			#clip_frames = torch.rand((32,3,crop_size_tsm,crop_size_tsm),dtype=torch.float32).to(device)

			print("Input: ")
			print(input_.shape)			
			input_ = torch.permute(input_, (0, 2, 1, 3, 4))
			print(input_.shape)
			input_ = input_.view(-1, 3, 224, 224)
			print(input_.shape)

			#exit()
			input_ = input_.to(device)
			net_tsm = net_tsm.to(device)
			output = net_tsm(input_)
			output = output.detach().cpu().numpy() 
			print(output.shape)

		elif feature_extractor == 'sshc':
			folder_index = 0	# Just used for buffer. Not used here
			sample_index = 0	# Just used for buffer. Not used here
			print("kkk")
			print(input_.shape)			
			adj_mat, bbox_fea_list, box_list, score_list = model.frames2temporalGraph(input_, folder_index, sample_index)
			print(bbox_fea_list)
			print("foi ;)")
			exit()

		feature_vector.append(output)

		if test_counter == 10:
			pass
			#break
		test_counter += 1
		
	feature_vector = np.stack(feature_vector, 0)

	if not ten_crop:
		feature_vector = feature_vector[:,0,:]

	print("Shape depois da rede")
	print(feature_vector.shape) # (T, 2048)




	return feature_vector


def checkValidVideo(video_path):
	
	mime = magic.Magic(mime=True)
	filename = mime.from_file(video_path)
	if filename.find('video') != -1:
		return True
	return False


def extractFrames(video, iteration, maxVideoFrame, ten_crop):
	# Opens the Video file
	cap= cv2.VideoCapture(video)
	limit = 99999999	# 30 frames * 20 segundos
	i = 0
	frames = []

	print("Iniciando o extract frames")
	print(iteration)
	print(maxVideoFrame)


	frameOffset = iteration*maxVideoFrame 
	#frameLimit = frameOffset + maxVideoFrame 
	cap.set(cv2.CAP_PROP_POS_FRAMES, frameOffset)
	while(cap.isOpened()):


		if i >= maxVideoFrame:
			break

		ret, frame = cap.read()
		if ret == 0:
			break

		i += 1

		try:
			# Resize here than in torchvision.transform because is more memory efficient. 
			#frame = cv2.resize(frame, dsize=(FRAMES_SIZE, FRAMES_SIZE), interpolation=cv2.INTER_CUBIC)
			frames.append(frame)
		
			#np.append(frames, frame)
			#if i > limit:
			#	break
		except Exception as e:
			print("Deu um pau ao redimensionar um dos frames")
			print(str(e))
	 
	cap.release()
	cv2.destroyAllWindows()	

	return frames


def countVideoFrames(path):

	path = os.path.join(path)
	cap = cv2.VideoCapture(path)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	return length



"""
""
"" Given the mp4 videos, we shall extract the feature vector, as made by RTFM, 
"" resunting in a npy files, for training phase
""
"""
def video2npy(video_path, path_to_save_npy, ten_crop, feature_extractor, gpu_id):


	# Check if file is a valid video
	if not checkValidVideo(os.path.join(video_path[0], video_path[1])):
		print("invalid video")
		return "invalid_video"

	#frames = extractFrames(video_path)
	#cutVideo(video_path)



	if ten_crop == True:
		print("Vamos relaizar o 10 crop")

		if feature_extractor == 'i3d':
			mean = [114.75, 114.75, 114.75]
			std = [57.375, 57.375, 57.375]			
			transform_norm = transforms.Compose([
				gtransformers.GroupTenCrop(TEN_CROP_SIZE),
				gtransformers.GroupTenNormalize(mean, std)
			])
		elif feature_extractor == 'c3d':
			mean = [104, 117, 128]
			std = [1]							
			transform_norm = transforms.Compose([
				gtransformers.GroupTenCrop(TEN_CROP_SIZE),
				gtransformers.GroupTenNormalize(mean, std)
			])
		elif feature_extractor == 'tsm' or feature_extractor == 'tsm-2048':
			net_tsm, crop_size_tsm, scale_size_tsm, input_mean_tsm, input_std_tsm = init_tsm_model(gpu_id)	

			transform_norm = transforms.Compose([
				# The default value of 10 crop is already 224, so the images will fit properly
				#gtransformers.GroupResize(224),	# Default dim of TSM code
				gtransformers.GroupTenCrop(TEN_CROP_SIZE),
				gtransformers.GroupNormalizeTSM(input_mean_tsm, input_std_tsm)
			])
		elif feature_extractor == "sshc":
			print("10 crop not implemented for SSHC")
			exit()
	else:
		print("NAO VAMOS REALIZAR O 10 CROP")

		if feature_extractor == 'i3d':
			mean = [114.75, 114.75, 114.75]
			std = [57.375, 57.375, 57.375]			
			transform_norm = transforms.Compose([
				gtransformers.GroupTenNormalize(mean, std)
			])

		elif feature_extractor == 'c3d':

			# Normalization as made here: https://github.com/jssprz/video_features_extractor/blob/master/feature_extractors/c3d.py
			mean = [104, 117, 128]
			std = [1]					
			# As made here: https://github.com/jssprz/video_features_extractor/blob/master/extract.py
			transform_norm = transforms.Compose([
				gtransformers.GroupResize(224),	# Default dim of TSM code
				gtransformers.GroupTenNormalize(mean, std)
			])

		elif feature_extractor == 'tsm' or feature_extractor == 'tsm-2048':
			#net_tsm, crop_size_tsm, scale_size_tsm, input_mean_tsm, input_std_tsm = init_tsm_model(gpu_id)			

			#print(input_mean_tsm)		
			#print(input_std_tsm)
			#exit()

			input_mean_tsm = [0.485, 0.456, 0.406]
			input_std_tsm = [0.229, 0.224, 0.225]	

			transform_norm = transforms.Compose([
				gtransformers.GroupResize(224),	# Default dim of TSM code
				gtransformers.GroupTenNormalize(input_mean_tsm, input_std_tsm)
			])
		elif feature_extractor == "sshc":
			transform_norm = transforms.Compose([
				#gtransformers.GroupResize(224),	# Default dim of TSM code
			])
			print("Ok, chegamos onde queríamos")
		


	model = chooseModel(feature_extractor, gpu_id)

	qtdFrames = countVideoFrames(os.path.join(video_path[0], video_path[1]))

	maxVideoFrame = 16000	# Multiplo de NUM_FRAMES_IN_EACH_FEATURE_VECTOR

	i = 0
	features_global = []
	for i in range(math.ceil(qtdFrames / maxVideoFrame)):

		frames = extractFrames(os.path.join(video_path[0], video_path[1]), i, maxVideoFrame, ten_crop)
		print("Oi")
		frames = np.asarray(frames)
		print(frames.shape)
		frames = np.stack(frames, 0)

		normalized = []
		#for i in range(frames.shape[0]):
		#print(frames[i].shape)
		#img = torch.from_numpy(frames[i])
		#print(img.shape)
		#img = torch.permute(img, (2, 0, 1))
		#print(img.shape)
		#img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
		frames = np.transpose(frames, (0, 3, 1, 2))
		frames = frames.astype('float16')
		frames = torch.from_numpy(frames)
		#frames = frames.float()


		# Aqui 'frames' é um tenor de inteiro

		img_normalized = transform_norm(frames) #.detach().cpu().numpy()

		#normalized = np.asarray(normalized)
		#normalized = np.stack(normalized, 0)

		#normalized = np.moveaxis(normalized, 1, 3)

		#print(normalized.shape)
		#exit()	

		#print("Shape dos dados que vão lá pro meu_main.py")
		#print(normalized.shape)


		# Agora video_path é um vídeo cortado e normalizado
		features = start(img_normalized, ten_crop, feature_extractor, gpu_id, model)
		features_global.extend(features)
		

	npy_name_file = video_path[1][:-4]+".npy"
	npy_name_file = path_to_save_npy + npy_name_file
	print(npy_name_file)

	print("Tamanho do vetor feature_vector: ")
	features_global = np.asarray(features_global)
	print(features_global.shape)
	np.save(npy_name_file, features_global)


		#classes = inferencia(features) 

		# Shoooow, temos as classes para cada segmento!!!

		#model = Model(args.feature_size, args.batch_size)




