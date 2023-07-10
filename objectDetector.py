
import numpy as np
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, mobilenet_backbone
from torchvision.models import resnet50
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps
import torch
import os
import sys
import definitions

class ObjectDetector:
	def __init__(self, device, training_obj_det = False, model="fasterrcnn_resnet50_fpn_coco"):

		self.model = None
		if model == "fasterrcnn_resnet50_fpn_coco":
			self.num_classes = 91
			self.what_backbone_model_use = "fasterrcnn_resnet50_fpn_coco"		
			self.device = device
			self.initBackboneModel()

			if not training_obj_det:
				self.freezeWeights()

		elif model == "yolov5":
			print("\n\n **************** Usando o YOLO como extrator de objetos\n\n")
			self.initBackBoneYolov5()
		

	def initBackboneModel(self):
		labels = ""

		# set the device we will be using to run the model

		# load the list of categories in the COCO dataset and then generate a
		# set of bounding box colors for each class
		#CLASSES = pickle.loads(open(labels, "rb").read())
		COLORS = np.random.uniform(0, 255, size=(self.num_classes, 3))


		model_urls = {
			'fasterrcnn_resnet50_fpn_coco':
				'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
			'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
				'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
			'fasterrcnn_mobilenet_v3_large_fpn_coco':
				'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
		}


		pretrained=True
		progress=True
		pretrained_backbone=True
		trainable_backbone_layers = None

		trainable_backbone_layers = _validate_trainable_layers(pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

		if pretrained:
			# no need to download the backbone if pretrained is set
			pretrained_backbone = False

		backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)

		self.model = FasterRCNN(backbone, self.num_classes)

		if pretrained:
			state_dict = load_state_dict_from_url(model_urls[self.what_backbone_model_use],
													progress=progress)
			self.model.load_state_dict(state_dict)
			overwrite_eps(self.model, 0.0)
		#self.model = self.model.cuda()
		self.model.eval()

		#image = loadImage() 
		#images = extractFrames()	

	def initBackBoneYolov5(self):
		YOLOV5_ROOT = os.path.join(definitions.ROOT_DIR, '../../../dependencies')
		print(YOLOV5_ROOT)
		if str(YOLOV5_ROOT) not in sys.path:
			print("porra")
			sys.path.append(str(YOLOV5_ROOT))  # add ROOT to PATH
			sys.path.append(str(YOLOV5_ROOT)+'/yolov5')  # add ROOT to PATH

		print(sys.path)
		from yolov5.utils.torch_utils import select_device
		from yolov5.models.common import DetectMultiBackend
		from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
								increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

	
		imgsz=(640, 640)

		weights = os.path.join(YOLOV5_ROOT, 'yolov5x.pt')
		dnn=False
		data = os.path.join(YOLOV5_ROOT, 'data/coco128.yaml')
		half = False
		bs = 1
		# Load model
		device = 'cpu'
		device = select_device(device)
		model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
		stride, names, pt = model.stride, model.names, model.pt
		imgsz = check_img_size(imgsz, s=stride)  # check image size		
		imgsz=(1 if pt else bs, 3, *imgsz)
		# Run inference
		model.warmup(imgsz=imgsz)  # warmup		

		self.model = model
		return


	def freezeWeights(self):
		ct = 0
		for child in self.model.children():
			for param in child.parameters():
				param.requires_grad = False

	def getModel(self):

		return self.model



