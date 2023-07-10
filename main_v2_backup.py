## Exemplo de: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Seems we do not need normalization for pytorch built-in models: https://github.com/pytorch/vision/blob/d2c763e14efe57e4bf3ebf916ec243ce8ce3315c/torchvision/models/detection/faster_rcnn.py#L227

# Explicação sobre Roi Polling e Roi Aling: 
# https://erdem.pl/pages/work

# Talvez um caminho de como acessar as features no RoiAlign:
#https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection

# Conferir esse feature extraction depois!
# https://github.com/pytorch/vision/pull/4302

import torch
from torchvision import transforms
import gtransformers


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
import objectDetector
import os
from definitions import ROOT_DIR, FRAMES_DIR, DATASET_DIR

import csv

import modelPretext

import modelDownstream
import datasetPretext
import datasetDownstream
import temporalGraph
import configparser
from util.utils import Visualizer
from util.utils import calculeTarget
from util.utils import print_image
from util.utils import calculeTargetAll
from util.utils import batch_processing
#import utils
from test import test
from test_downstream import test as test_downstream
import losses

viz = Visualizer(env='Graph_Detector', use_incoming_socket=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    


config = configparser.ConfigParser(allow_no_value=True)
config.sections()
config.read('config.ini')
T = int(config['PARAMS']['T'])                                      # Frame window. The object predict is always in the last frame
N = int(config['PARAMS']['N'])                                      # How many objects we will consider for each frame?
LOOK_FORWARD = int(config['PARAMS']['LOOK_FORWARD'])
STRIDE = int(config['PARAMS']['STRIDE'])                            # STRIDE for each sample
MAX_EPOCH = int(config['PARAMS']['MAX_EPOCH'])                      # Training max epoch
LR = float(config['PARAMS']['LR'])                                    # Learning rate
OBJECT_FEATURE_SIZE = int(config['PARAMS']['OBJECT_FEATURE_SIZE'])  # OBJECT_FEATURE_SIZE
SIMILARITY_THRESHOLD = float(config['PARAMS']['SIMILARITY_THRESHOLD'])
BBOX_FEATURES = int(config['PARAMS']['BBOX_FEATURES'])

GT_PATH = os.path.join(ROOT_DIR, '../')

# Allow only 1 (person) 2 (bicycle) 3 (car) 4 (motorcycle)
OBJECTS_ALLOWED = [1,2,3,4]    # COCO categories ID allowed. The othwers will be discarded


FEA_DIM_IN = 0
FEA_DIM_OUT = 0

OUTPUT_PATH_PRETEXT_TASK = os.path.join(ROOT_DIR, "results/i3d/pretext_task")
OUTPUT_PATH_DOWNSTREAM_TASK = os.path.join(ROOT_DIR, "results/i3d/downstream_task")
MODEL_NAME = "model"

# Exemplo de como trabalhar com hook: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def loadImage():
    #image = PIL.Image.open("zidane.jpeg").convert('RGB').convert('RGB')
    image = cv2.imread("teste.png")

    return image

def train(save_folder):

    global SIMILARITY_THRESHOLD, MAX_EPOCH, T, LOOK_FORWARD
    print("Iniciando treinamento para")
    print("T = " + str(T) + "; N = " + str(N) + "; LR = " + str(LR) + "; STRIDE: "+str(STRIDE)+"; SIMILARITY_THRESHOLD: " + str(SIMILARITY_THRESHOLD)) 

    training_loss_log = os.path.join(save_folder, "training_log.txt")
    test_loss_log = os.path.join(save_folder, "test_log.txt")
    trining_log = open(training_loss_log, 'a')
    test_log = open(test_loss_log, 'a')

    buffer_size = T*5
    temporal_graph = temporalGraph.TemporalGraph(DEVICE, buffer_size, OBJECTS_ALLOWED, N, STRIDE, BBOX_FEATURES=BBOX_FEATURES)
    #temporal_graph.generateTemporalGraph()

    batch_size = 1              # Aumentar no final
    max_sample_duration = 200   # Limitando as amostras por no máximo 200 arquivos.png
    # each video is a folder number-nammed
    training_folder = os.path.join(FRAMES_DIR, "training")
    print(training_folder)

    train_loader = DataLoader(datasetPretext.DatasetPretext(LOOK_FORWARD, STRIDE, training_folder, max_sample_duration),
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)   

    test_folder = os.path.join(FRAMES_DIR, "test")
    test_loader = DataLoader(datasetPretext.DatasetPretext(LOOK_FORWARD, STRIDE, test_folder, max_sample_duration, test=True),
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)   

    # I3D Input: (1, 3, 16, 224, 224), 16 is the segment lenght
    model_used = "i3d"
    model = modelPretext.ModelPretext(FEA_DIM_IN, FEA_DIM_OUT, model_used).chooseModel().to(DEVICE)
    #model = modelPretextTransformers.modelPretextTransformers(FEA_DIM_IN, FEA_DIM_OUT).to(DEVICE)
    print(model)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=LR, weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}

    # index of object in the object listto be predicted
    obj_predicted = 0
    reference_frame = 0

    best_loss = float("+Inf")    
    loss_mean = test(model, model_used, loss, test_loader, reference_frame, obj_predicted, viz, buffer_size, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, LOOK_FORWARD, OBJECTS_ALLOWED, STRIDE)    
    test_log.write(str(loss_mean) + " ")
    test_log.flush()
    torch.save(model.state_dict(), os.path.join(save_folder, MODEL_NAME + '{}.pkl'.format(0)))

    data_loader = iter(train_loader)    
    max_epoch_ = len(data_loader) * MAX_EPOCH
    t = tqdm(
            range(1, max_epoch_ + 1),
            total=max_epoch_,
            dynamic_ncols=True
    )
    t.refresh()  #force print final state
    t.reset()  #reuse bar    
    for step in range(len(t)):
        t.update()

        with torch.set_grad_enabled(True):
            model.train()

            print("Quantidade de amostras no datase: " + str(len(data_loader)))
            if (step - 1) % len(data_loader) == 0:
                data_loader = iter(train_loader)

            # input: [T, W, H, C]
            input = next(data_loader)
            # Attention! This works only when batch_size = 1
            folder_index = input[1][0]
            sample_index = input[2][0]
            input = input[0]

            if model_used == 'i3d':
                # I3D recieves: (1, 3, 16, 224, 224), 16 is the segment lenght. So we have to adjust
                shape_ = input.shape
                input_frames = input.view(shape_[0], shape_[4], shape_[1], shape_[2], shape_[3]).type(torch.FloatTensor).to(DEVICE)

                # In the pretext task, we have to predict a future frame, so we need here just the first T frames. After, we will predicrt the LOOK_FORWARDº frame
                input_frames = input_frames[:, :, 0:T, :, :]            

                mean = [114.75, 114.75, 114.75]
                std = [57.375, 57.375, 57.375]			
                transform_norm = transforms.Compose([
                    gtransformers.GroupTenNormalize(mean, std)
                ])

                input_frames = transform_norm(input_frames) #.detach().cpu().numpy()

            input = np.squeeze(input)

            # Returns [T-1, obj1, obj2], beeing obj1 the num object detected in the first frame and obj2 in the second frame
            # [] if a frame does not have objects

            if model_used == 'i3d':
                T = LOOK_FORWARD
            cache_folder = "cache_pt_task/i3d/train/T="+str(T)+"-N="+str(N)+"/"
            data_path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()), str(sample_index.cpu().numpy())+"_data.npy")
            print(data_path)
            has_cache = False
            if os.path.exists(data_path):
                #train_loader.has_cache = True
                data_loader.has_cache = True
                has_cache = True
            else:
                has_cache = False

            if not has_cache:
                adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(input, folder_index, sample_index)
                #SIMILARITY_THRESHOLD = 0.65#0.73       # Resnet with fea vector
                #SIMILARITY_THRESHOLD = 0.97#0.73
                graph = calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, SIMILARITY_THRESHOLD, T, N)

                # If in the first frame there is no object detected, so we have nothing to do here
                # The number of detected objects may be less than N. In this case we have nothing to do here
                #if len(bbox_fea_list[reference_frame][obj_predicted]) < N:
                #    print("continuando")
                #    continue       # Continue

                #data, object_path = calculeTarget(adj_mat, bbox_fea_list, box_list, reference_frame, obj_predicted, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)
                data, object_path = calculeTarget(graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

                path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()))
                os.makedirs(path, exist_ok=True)

                if data != -1:
                    data = [data[0].cpu().numpy(), data[1].cpu().numpy()]

                print(data)
                np.save(data_path, data)

            else:
                print("Ok, temos cache, vamos carregar")
                data = np.load(data_path, allow_pickle=True)

            if data != -1:
                print("Continuing because there aren't a object in the first frame ")
                #continue

                #print("\n\nPRINTANDO IMAGEM!!@!!\n\n")
                #print_image(input, box_list, object_path, step)

                data = [torch.from_numpy(data[0]).to(DEVICE), torch.from_numpy(data[1]).to(DEVICE)]
                input, target = data       
                input = input.to(DEVICE)
                target = target.to(DEVICE)
                if model_used == 'i3d':
                    print("POOOOOOOOORRA")
                    print(input_frames.shape)
                    input_frames = {'frames': input_frames}
                    output = model(input_frames)
                else:
                    output = model(input)

                print(output.shape)    

                loss_ = loss(output, target)
                #viz.plot_lines('loss', cost.item())
                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()

                #viz.plot_lines('training_loss', loss_.item())

                trining_log.write(str(loss_.item()) + " ")

                print("\n\nStep!!!!!!!!!!: " + str(step))
                #print("data loader: " + str(len(data_loader)))

            if step % len(data_loader) == 0 and step > 1:
                trining_log.flush()
                loss_mean = test(model, model_used, loss, test_loader, reference_frame, obj_predicted, viz, buffer_size, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, LOOK_FORWARD, OBJECTS_ALLOWED, STRIDE)    
                test_log.write(str(loss_mean) + " ")
                test_log.flush()

                if loss_mean < best_loss:
                    # Save model 
                    torch.save(model.state_dict(), os.path.join(save_folder, MODEL_NAME + '{}.pkl'.format(step)))                    
                    fo = open(os.path.join(save_folder, MODEL_NAME + '{}.txt'.format(step)), "w")
                    fo.write("Test loss: " + str(loss_mean))
                    fo.close()     
                    best_loss = loss_mean    


    trining_log.close()
    test_log.close()           

def run():
    global T, N, LOOK_FORWARD, LR, FEA_DIM_IN, OBJECT_FEATURE_SIZE, FEA_DIM_OUT, SIMILARITY_THRESHOLD, EXIT_TOKEN

    #T_ = [T, T-1, T-2, T-3]
    T_ = [T]
    N_ = [N, N-1, N-2, N-3, N-4]
    #LR_ = [LR*10, LR, LR/10]
    #SIMILARITY_THRESHOLD_ = [SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD-0.1, SIMILARITY_THRESHOLD-0.2]


    for t in T_:
        for n in N_:
            for u in range(5):  # The training is unstable. We have to train 3x and get the greater result
                #for lr in LR_:
                #for st in SIMILARITY_THRESHOLD_:
                T = t
                N = n
                #LR = lr
                #SIMILARITY_THRESHOLD = st            # Threshold to verify if two detected are the same

                FEA_DIM_IN = (OBJECT_FEATURE_SIZE * N * (T-1)) + (4 * N * (T-1))
                FEA_DIM_OUT = OBJECT_FEATURE_SIZE + 4
                EXIT_TOKEN = FEA_DIM_OUT

                #if (T == 2 and N == 1 and LR == 0.005) or (T == 2 and N == 1 and LR == 0.0005) or (T == 2 and N == 1 and LR == 0.00005):
                #    if st == 0.7:
                #        continue
                #if (T == 2 and N == 2 and LR == 0.0005):
                #    if st == 0.7 or st == 0.6 or st == 0.5:
                #        continue


                save_folder = "t="+str(T)+"-n="+str(N)+"-lr="+str(LR)+"-st="+str(SIMILARITY_THRESHOLD)+"-"+str(u)
                save_folder = os.path.join(OUTPUT_PATH_PRETEXT_TASK, save_folder)

                try:
                    os.mkdir(save_folder)
                except OSError as error:
                    print("Erro ao criar dir: ")
                    print(error)    
                    continue

                train(save_folder)


def downstreamTask(T, N, st, N_DOWNSTRAM, FEA_DIM_IN, FEA_DIM_OUT, pretext_checkpoint, downstream_folder, checkpoint):
    #global FEA_DIM_OUT, FEA_DIM_OUT

    EXIT_TOKEN = FEA_DIM_OUT

    trining_log = open(os.path.join(downstream_folder, "training_log.txt"), 'a')
    test_log = open(os.path.join(downstream_folder, "test_log.txt"), 'a')

    print("Carregando o checkpoint ")
    print(pretext_checkpoint)

    #RANDOM_WEIGHTS =0
    #if RANDOM_WEIGHTS == 1:
    #    print("ATENÇÃO, VC ESTÁ CARREGANDO UM PRETEXT MODEL RANDOMIZADO")
    #    pretext_checkpoint = os.path.join(ROOT_DIR, "results/pretext_task/t=5-n=5-lr=5e-05-st=0.7_RANDOM_WEIGHTS/model_random_weights.pkl")

    model_pt = modelPretext.ModelPretext(FEA_DIM_IN, FEA_DIM_OUT)
    model_pt.load_state_dict(torch.load(pretext_checkpoint))

    batch_size = 12800 #200
    STRIDE = T
    # We use two temporal graph instances because we have an individual buffer to save computational power
    temporal_graph_normal = temporalGraph.TemporalGraph(DEVICE, batch_size, OBJECTS_ALLOWED, N, STRIDE)
    temporal_graph_abnormal = temporalGraph.TemporalGraph(DEVICE, batch_size, OBJECTS_ALLOWED, N, STRIDE)

    #model_pt.ModelPretext = nn.Sequential(*list(model_pt.ModelPretext.children())[:-1])
    prunned_model_pt = nn.Sequential(*list(model_pt.children())[:-2])
    prunned_model_pt.train()
    
    # A entrada vai ser o tamanho da penútima saída do Pretext Model
    model = modelDownstream.ModelDownstream(128).to(DEVICE)
    ct = 0
    # Make all model trainable
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True    

    LR_DOWNSTREAM = 0.00005

    max_sample_duration = 300
    normal_dataset = DataLoader(datasetDownstream.DatasetDownstream(T, max_sample_duration, normal = True, test=False), batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)

    abnormal_dataset = DataLoader(datasetDownstream.DatasetDownstream(T, max_sample_duration, normal = False, test=False), batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)

    list_ = os.path.join(ROOT_DIR, "../", "files/graph_detector_test_05s.list")
    print("listy")
    print(list_)
    test_dataset = DataLoader(datasetDownstream.DatasetDownstream(T, max_sample_duration, list_=list_, normal = True, test=True), batch_size=1, shuffle=False,
                                   num_workers=0, pin_memory=False)


    optimizer = optim.Adam(model.parameters(),
                            lr=LR_DOWNSTREAM, weight_decay=0.005)

    STRIDE_TEST = 1     #
    if checkpoint is not False:
        # If we are loading a pre-trained model, we don't need retest it in the start
        print("Continuando o treinamento do modelo: ")
        model.load_state_dict(torch.load(checkpoint))
        start_epoch = int(checkpoint.split('model')[1].split('.pkl')[0]) + 1
        print("Começando de " + str(start_epoch))
    else:
        # TODO: BETTER WAY TO PROPAGATE THE STRIDE TO DATASETDOWNSTREAM
        print("Comançando o treinamento do zero")
        auc = test_downstream(test_dataset, prunned_model_pt, model, viz, max_sample_duration, list_, STRIDE_TEST, DEVICE, False, GT_PATH, OBJECTS_ALLOWED, N, T, EXIT_TOKEN)
        test_log.write(str(auc) + " ")
        test_log.flush()
        best_auc = auc
        step = 0
        torch.save(model.state_dict(), os.path.join(downstream_folder, MODEL_NAME + '{}.pkl'.format(step)))                    
        fo = open(os.path.join(downstream_folder, MODEL_NAME + '{}.txt'.format(step)), "w")
        fo.write("Test loss: " + str(best_auc))
        fo.close()     
        start_epoch = 1

    normal_loader = iter(normal_dataset)    
    abnormal_loader = iter(abnormal_dataset)    
    
    for step in tqdm(
            range(start_epoch, MAX_EPOCH + 1),
            total=MAX_EPOCH,
            dynamic_ncols=True
    ):    

        with torch.set_grad_enabled(True):
            model.train()
            prunned_model_pt.train()

            if (step - 1) % len(normal_loader) == 0:
                normal_loader = iter(normal_dataset)  

            if (step - 1) % len(abnormal_loader) == 0:
                abnormal_loader = iter(abnormal_dataset)

            # [[sample, label, folder_index, sample_index] ...]
            input_normal = next(normal_loader)
            input_abnormal = next(abnormal_loader)

            batch_list = batch_processing(input_abnormal, input_normal, temporal_graph_normal, temporal_graph_abnormal, normal_loader, abnormal_loader, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

            input_abnormal = [i[0] for i in batch_list]
            input_normal = [i[1] for i in batch_list]

            if len(input_abnormal) == 0 or len(input_normal) == 0:
                print("Alguma amostra possui objetos no primeiro frame")
                continue

            input_abnormal = torch.stack(input_abnormal)
            input_normal = torch.stack(input_normal)

            abnormal_res = model(prunned_model_pt(input_abnormal))
            normal_res = model(prunned_model_pt(input_normal))

            downstramLoss = losses.DownstramLoss(normal_res, abnormal_res)
            cost = downstramLoss()
            print("loss: ")
            print(cost)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            trining_log.write(str(cost.item()) + " ")
            print("Step: ")
            print(step)
            print(len(normal_loader))
            if step % len(normal_loader) == 0:# and step > 10:
            #if step % 5 == 0:
                trining_log.flush()

                auc = test_downstream(test_dataset, prunned_model_pt, model, viz, max_sample_duration, list_, STRIDE_TEST, DEVICE, False, GT_PATH, OBJECTS_ALLOWED, N, T, EXIT_TOKEN)
                #loss_mean = test(model, loss, test_loader, reference_frame, obj_predicted, viz, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, OBJECTS_ALLOWED)    
                test_log.write(str(auc) + " ")
                test_log.flush()

                if auc > best_auc:
                    # Save model 
                    torch.save(model.state_dict(), os.path.join(downstream_folder, MODEL_NAME + '{}.pkl'.format(step)))                    
                    print("Saving model at")
                    print(os.path.join(downstream_folder, MODEL_NAME + '{}.pkl'.format(step)))
                    fo = open(os.path.join(downstream_folder, MODEL_NAME + '{}.txt'.format(step)), "w")
                    fo.write("Test loss: " + str(auc))
                    fo.close()     
                    best_auc = auc              
    test_log.close()
    trining_log.close()

def runDownstream():

    checkpoint = False
    #checkpoint = os.path.join(ROOT_DIR, "results/downstream_task/t=5-n=5-lr=5e-05-st=0.7/model72.pkl")

    config = configparser.ConfigParser(allow_no_value=True)
    config.sections()
    config.read('config.ini')
    T = int(config['PARAMS']['T'])                                      # Frame window. The object predict is always in the last frame
    N = int(config['PARAMS']['N'])                                      # How many objects we will consider for each frame?
    STRIDE = int(config['PARAMS']['STRIDE'])                            # STRIDE for each sample
    MAX_EPOCH = int(config['PARAMS']['MAX_EPOCH'])                      # Training max epoch
    LR = float(config['PARAMS']['LR'])                                  # Learning rate
    OBJECT_FEATURE_SIZE = int(config['PARAMS']['OBJECT_FEATURE_SIZE'])  # OBJECT_FEATURE_SIZE
    SIMILARITY_THRESHOLD = float(config['PARAMS']['SIMILARITY_THRESHOLD'])

    FEA_DIM_IN = (OBJECT_FEATURE_SIZE * N * (T-1)) + (4 * N * (T-1))
    FEA_DIM_OUT = OBJECT_FEATURE_SIZE + 4
    EXIT_TOKEN = FEA_DIM_OUT

    #N_DOWNSTRAM = N         # Tamanho da janela para ver se é normal ou abnormal
    

    #pretext_path = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/graph_detector/results/pretext_task/"


    T_ = [T, T+1, T+2, T+3]
    N_ = [N, N+1, N+2]
    #LR_ = [LR*10, LR, LR/10]
    SIMILARITY_THRESHOLD_ = [SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD-0.1, SIMILARITY_THRESHOLD-0.2]

    for t in T_:
        for n in N_:

            #for lr in LR_:
            for st in SIMILARITY_THRESHOLD_:      

                N_DOWNSTRAM = n
                pretext_folder_sufix = "t="+str(t)+"-n="+str(n)+"-lr="+str(LR)+"-st="+str(st)
                pretext_folder = os.path.join(OUTPUT_PATH_PRETEXT_TASK, pretext_folder_sufix)                    
                pretext_checkpoint = os.path.join(pretext_folder, find_value(pretext_folder))          
                
                downstream_folder = os.path.join(OUTPUT_PATH_DOWNSTREAM_TASK, pretext_folder_sufix)
                #if we are continuing a training
                if checkpoint is not False:
                    print("Continuando o treinamento do modelo: ")
                    # Verify if the checkpoint is in the same folder than  downstream_folder
                    assert(os.path.dirname(checkpoint) == downstream_folder)
                else:

                    try:
                        os.mkdir(downstream_folder)
                    except OSError as error:
                        print("Erro ao criar dir: ")
                        print(error)    
                        continue
                
                downstreamTask(t, n, st, N_DOWNSTRAM, FEA_DIM_IN, FEA_DIM_OUT, pretext_checkpoint, downstream_folder, checkpoint)


def find_value(dir):

    #return ''
    score = []
    files = []
    # Find all files with model*.txt in the 'dir' folder
    print(dir)
    for file in os.listdir(dir):
        if 'model' in file and file.endswith(".txt"):
            number = file[5:][:-4]
            files.append(int(number))

    files.sort()
    the_one = files[-1]

    the_one = 0
    nome = "model"+str(the_one)+".pkl"
    return nome

if __name__ == '__main__':

    # Search training parameter
    run()
    #downstreamTask()
    #runDownstream()




            


