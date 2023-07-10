import torch
import torch.nn as nn
import torch.nn.init as torch_init


# TODO:
# Weight init
class ModelDownstream(nn.Module):
	def __init__(self, num_feat_in):
		super(ModelDownstream, self).__init__()	
		print("num_feat_in: ")
		print(num_feat_in)
	
		self.fc1 = nn.Linear(num_feat_in, 512)
		self.fc2 = nn.Linear(512, 1024)
		self.fc3 = nn.Linear(1024, 1)     

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid() 

	def forward(self, inputs):
		fc1 = self.relu(self.fc1(inputs))
		fc2 = self.relu(self.fc2(fc1))
		fc3 = self.sigmoid(self.fc3(fc2))

		return fc3
