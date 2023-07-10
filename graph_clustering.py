## Exemplo de: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Seems we do not need normalization for pytorch built-in models: https://github.com/pytorch/vision/blob/d2c763e14efe57e4bf3ebf916ec243ce8ce3315c/torchvision/models/detection/faster_rcnn.py#L227

# Explicação sobre Roi Polling e Roi Aling: 
# https://erdem.pl/pages/work

# Talvez um caminho de como acessar as features no RoiAlign:
#https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection

# Conferir esse feature extraction depois!
# https://github.com/pytorch/vision/pull/4302

import torch

import numpy as np
import PIL
import pickle
import cv2
import time
import torch.nn as nn
from typing import Dict, Iterable, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
#import objectDetector
import os
import csv
from sklearn.cluster import AgglomerativeClustering


import datasetClustering
import temporalGraph
import configparser
from utils import Visualizer
#from utils import calculeTarget
#from utils import print_image
import utils
#from test import test
#from test_downstream import test as test_downstream
#import losses

#viz = Visualizer(env='Graph_Detector', use_incoming_socket=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    


config = configparser.ConfigParser(allow_no_value=True)
config.sections()
config.read('config.ini')
C = int(config['GRAPH_CLUSTERING']['C'])                                      # Frame window. The object predict is always in the last frame
N_DOWNSTRAM = 5
OBJECTS_ALLOWED = [1,2,3,4]    # COCO categories ID allowed. The othwers will be discarded
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = int(config['PARAMS']['T'])                                      # Frame window. The object predict is always in the last frame
N = int(config['PARAMS']['N'])                                      # How many objects we will consider for each frame?
STRIDE = int(config['PARAMS']['STRIDE'])                            # STRIDE for each sample
MAX_EPOCH = int(config['PARAMS']['MAX_EPOCH'])                      # Training max epoch
LR = float(config['PARAMS']['LR'])                                    # Learning rate
OBJECT_FEATURE_SIZE = int(config['PARAMS']['OBJECT_FEATURE_SIZE'])  # OBJECT_FEATURE_SIZE
SIMILARITY_THRESHOLD = float(config['PARAMS']['SIMILARITY_THRESHOLD'])

FEA_DIM_IN = (OBJECT_FEATURE_SIZE * N) + (4 * N)
FEA_DIM_OUT = OBJECT_FEATURE_SIZE + 4
EXIT_TOKEN = FEA_DIM_OUT

def make_obj_graph(object_path, bbox_fea_list, box_list):


    print(len(object_path))
    print(len(bbox_fea_list))
    
    x = []
    for i in range(len(object_path)-1):
        index = int(object_path[i])
        if index == -1:
            break
        x.append(bbox_fea_list[i][0][index])
    
    index = int(object_path[i])
    x.append(bbox_fea_list[i][1][index])

    clustering = clustering = AgglomerativeClustering(n_clusters=3).fit(x)
    print(clustering.labels_)

    return clustering    



temporal_graph = temporalGraph.TemporalGraph(DEVICE, OBJECTS_ALLOWED, N_DOWNSTRAM)

batch_size = 1
train_dataset = DataLoader(datasetClustering.DatasetClustering(C),
                               batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=False) 
train_loader = iter(train_dataset)

reference_frame = 0
for i in range(1):
    sample = next(train_loader)
    sample = next(train_loader)
    sample = next(train_loader)
    sample = next(train_loader)
    sample = next(train_loader)
    sample = next(train_loader)
    sample = next(train_loader)  
    sample = next(train_loader)
    
    print(sample.shape)
    sample = torch.squeeze(sample)

    #input_normal, box_list_normal = utils.img2bbox([sample], temporal_graph, N_DOWNSTRAM)    

    #print(input_normal.shape)    

    adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(sample)    

    print(adj_mat[0].shape)

    # graph is [N,T] matrix, contening the path of every object through frames
    SIMILARITY_THRESHOLD = 0.65#0.73
    graph = utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, C, N_DOWNSTRAM)
    print("Len do graph: ")
    print(len(graph))

    object_list = []
    # For each object
    cont = 0
    last_index = 0
    for f in range(len(score_list)-1):  # Total frame qtt 

        print(len(graph[f]))
        print(len(object_list))
        if len(graph[f]) > len(object_list):
            object_list = graph[f]
        else:
            #object_list = []
            continue

        print("Qtd elementos no graph")
        print(object_list)

        for index_obj in range(len(object_list[last_index:])):     # Each frame can has different number of objects

            # Given the path graph of every object of every frame, make a single path graph of 
            # the 'obj' object starting at 'reference frame'
            
            object_path = utils.calculeObjectPath(graph, f, index_obj+last_index)

            #clustering = make_obj_graph(object_path, bbox_fea_list, box_list)

            utils.print_image(sample[f:], box_list[f:], object_path, cont)

            cont += 1

        last_index += index_obj+1



