mean = [114.75, 114.75, 114.75]
std = [57.375, 57.375, 57.375]

split == '10_crop_ucf':
transform = transforms.Compose([
gtransforms.GroupResize(256),
gtransforms.GroupTenCrop(224),
gtransforms.ten_crop_ToTensor(),
gtransforms.GroupNormalize_ten_crop(mean, std),
gtransforms.LoopPad(max_len),
])

class GroupTenCrop(object):
	def init(self, size):
		transform = torchvision.transforms.Compose([
		torchvision.transforms.TenCrop(size),
		torchvision.transforms.Lambda(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
		])
		
		self.worker = transform
	
	def call(self, img_group):
		return [self.worker(img) for img in img_group]

class ToTensor(object):
	def init(self):
		self.worker = lambda x: F.to_tensor(x) * 255
	
	def call(self, img_group):
		img_group = [self.worker(img) for img in img_group]
		return torch.stack(img_group, 0)