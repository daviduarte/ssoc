import numpy as np
import torch
import torchvision
from PIL import Image


TEN_CROP_SIZE = 224
"""
Util functions
"""
def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

"""
Transformations Classes
"""
class GroupTenCrop(object):
    def __init__(self, size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.TenCrop(size),
            # The line bellow (from https://pytorch.org/vision/master/generated/torchvision.transforms.TenCrop.html) 
            # is not working. Maybe by my torch version?
            #torchvision.transforms.Lambda(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
        ])

        self.worker = transform

    def __call__(self, img_group):

        
        group = []
        for img in img_group:
            
            ten_cropped = tuple_of_tensors_to_tensor(self.worker(img))
            group.append(ten_cropped)

        return torch.stack(group, 0)

class GroupTenCropToTensor(object):
    def __init__(self):
        self.worker = lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) * 255 for crop in crops])
    
    def __call__(self, crops):
        group_ = [self.worker(crop) for crop in crops]
        stack = torch.stack(group_, 1)
        return stack        

class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize((size, size), interpolation)

    def __call__(self, img_group):

        group_ = [self.worker(img) for img in img_group]
        stack = torch.stack(group_, 0)
        print(stack.shape)
        
        return stack


class ToTensor(object):
    def __init__(self):
        #transform = torchvision.transforms.Compose([
        #    torchvision.transforms.TenCrop(size)
        #    #torchvision.transforms.Lambda(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
        #])        
        self.worker = lambda x: torch.from_numpy(x)
        
    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        oi  = torch.stack(img_group, 0)

        return oi

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor): # (T, 3, 224, 224)
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor

# Normalization got from TSM original code
class GroupNormalizeTSM(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

class GroupTenNormalize(object):
  def __init__(self, mean, std):
      self.worker = GroupNormalize(mean, std)
  
  def __call__(self, crops):    
      
      group_ = [self.worker(crop) for crop in crops]
      stack = torch.stack(group_, 0)
      return stack