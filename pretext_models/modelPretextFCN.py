import torch
import torch.nn as nn
import torch.nn.init as torch_init


# TODO:
# Weight init
class ModelPretextFCN(nn.Module):
	def __init__(self, num_feat_in, num_feat_out):
		super(ModelPretextFCN, self).__init__()	

		print("Utilizando FCN como camadas transferíveis")
		self.fc1 = nn.Linear(num_feat_in, 1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 128)
		self.fc4 = nn.Linear(128, num_feat_out)   

		self.relu = nn.ReLU()  

	def forward(self, inputs):	# Input é o ROI Align da Resnet. Não está limitado a 255. Range real é: [0, inf]

		fc1 = self.relu(self.fc1(inputs))
		fc2 = self.relu(self.fc2(fc1))
		fc3 = self.relu(self.fc3(fc2))

		# In the downstream, the target is the object resnet feature vector. This is in the range [0-255] because is uint8
		# So, relu may do the work in the last layer		
		# Hovever, in the training of P.T, w the target is a RoiAlign from Resnet, which value range is [0,Inf]. So, relu keeps doing the work
		fc4 = self.relu(self.fc4(fc3))	

		return fc4
