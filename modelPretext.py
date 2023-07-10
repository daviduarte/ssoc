from pretext_models import modelPretextFCN
import os

from definitions import ROOT_DIR
#from pretext_models import modelPretextYolov5
import sys
print("Root dir printado no pretextModwel.py")
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "../../extractI3d/pytorch-resnet3d/models"))
import resnet
from other_models import resnetHead

# TODO:
# Weight init
class ModelPretext():
	def __init__(self, num_feat_in, num_feat_out, model):

		self.num_feat_in = num_feat_in
		self.num_feat_out = num_feat_out

		self.model = model

	def chooseModel(self):
		if self.model == "FCN":
			return modelPretextFCN.ModelPretextFCN(self.num_feat_in, self.num_feat_out)
		elif self.model == "i3d":
			print("Dimensões das feat out: ")

			resnet_ = resnet.i3_res50(400) # vanilla I3D ResNet50
			ResnetHead = resnetHead.resnetHead(2048, self.num_feat_out, resnet_)		# Feat size

			return ResnetHead

		#elif self.model == "yolov5":
		#	return modelPretextYolov5.ModelPretextYolov5(self.num_feat_in, self.num_feat_out)


