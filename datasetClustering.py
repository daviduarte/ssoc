import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import cv2
import random
import math

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class DatasetClustering(data.Dataset):
    #def __init__(self, args, is_normal=True, transform=None, test_mode=False, only_anomaly=False):
    def __init__(self, C):
            self.C = C
            self.frame_folders = {}
            self.training_folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/pretext_task_teste/05s/training"
            self.totalSample = 0
            self._parse_list()

    def _parse_list(self):
        self.frame_folders['list'] = []
        self.frame_folders['sample_num'] = []

        for filename in os.listdir(self.training_folder):

            frame_folder_path = os.path.join(self.training_folder, filename)

            if os.path.isdir(frame_folder_path):
                self.frame_folders['list'].append(frame_folder_path)     
                qtd = self.countFiles(frame_folder_path, 'png')
                qtd = math.ceil(qtd / self.C)
                self.frame_folders['sample_num'].append(qtd)    

            self.totalSample += 1   

    def __len__(self):
        return self.totalSample                

    def __getitem__(self, index):      

        sample = []
        index += 1

        folder = 0

        cont = 0

        if index > 1:
            while cont+self.frame_folders['sample_num'][folder] < index:
                cont += self.frame_folders['sample_num'][folder]
                folder += 1

            folder -= 1

        in_folder = index - cont
        frames_in_folder = self.countFiles(self.frame_folders['list'][folder], 'png')
     
        # Lê as amostras
        for i in range((in_folder-1)*self.C+1, (in_folder-1)*self.C + self.C + 1):
            
            if i > frames_in_folder:
                break

            pathSample = os.path.join(self.frame_folders['list'][folder], str(i)+'.png')                

            img = cv2.imread(pathSample)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                   
            sample.append(img)
        


        sample = np.stack(sample, axis=0)
        return sample

    def countFiles(self, path, extension):
        counter = 0
        for img in os.listdir(path):
            ex = img.lower().rpartition('.')[-1]

            if ex == extension:
                counter += 1
        return counter            