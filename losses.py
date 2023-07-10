import numpy as np
import torch



class DownstramLoss(torch.nn.Module):
	def __init__(self, normal_res, abnormal_res):
		super(DownstramLoss, self).__init__()

		self.bigger_anomalous = torch.max(abnormal_res)
		self.bigger_normal = torch.max(normal_res)


	def forward(self):
		#print("Maior anômalo: ")
		#print(self.bigger_anomalous)
		#print("Maior normal: ")
		#print(self.bigger_normal)
		return 1 - (self.bigger_anomalous - self.bigger_normal)