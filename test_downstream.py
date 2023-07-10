import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import util.utils as utils
import temporalGraph
import os
import cv2
import math
from definitions import DATASET_DIR, FRAMES_DIR

def getFrameQtd(frame_folder_path):
    video_path = frame_folder_path.replace('CamNuvem_dataset_normalizado_frames_05s', 'CamNuvem_dataset_normalizado/videos/samples')
    video_path = video_path.replace('/10', '/10.mp4')

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


#param labels A txt file path containing all test/anomaly frame level labels
#param list A txt file path containing all absolut path of every test file (normal and anomaly)
def getLabels(labels, list_test):

    # Colocar isso no config.ini depois
    # TODO
    test_normal_folder = os.path.join(DATASET_DIR, "videos/samples/test/normal")
    test_anomaly_folder = os.path.join(DATASET_DIR, "videos/samples/test/anomaly")

    with open(labels) as file:
        lines = file.readlines()
    qtd_anomaly_files = len(lines)

    gt = []
    qtd_total_frame = 0
    anomaly_qtd = 0
    for line in lines:        

        line = line.strip()
        list = line.split("  ")

        video_name = list[0]
        video_path = os.path.join(test_anomaly_folder, video_name)
        
        # First we create an array with 'frame_qtd' zeros
        # Zeros represents the 
        cap = cv2.VideoCapture(video_path)
        frame_qtd = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        qtd_total_frame += frame_qtd

        frame_label = np.zeros(frame_qtd)

        labels = list[1]
        labels = labels.split(' ')

        assert(len(labels) % 2 == 0) # We don't want incorrect labels
        sample_qtd = int(len(labels)/2)
        
        
        for i in range(sample_qtd):
            index = i*2
            start = round(float(labels[index]) * frame_qtd)
            end = round(float(labels[index+1]) * frame_qtd)
            
            frame_label[start:end] = 1

        gt.append([video_name, frame_label])

        anomaly_qtd += 1




    #############################################################

    lines = []
    with open(list_test) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]


    list_ = []
    cont = 0
    for path in lines:
        cont+=1
        if cont <= anomaly_qtd:
            continue
        filename = os.path.basename(path)  
        list_.append(os.path.join(test_normal_folder, filename[:-4]+'.mp4'))


    # Lets get the normal videos
    qtd_total_frame = 0
    for video_path in list_:
        video_path = video_path.strip()

        # First we create an array with 'frame_qtd' zeros
        # Zeros represents the 
        
        cap = cv2.VideoCapture(video_path)
        frame_qtd = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        qtd_total_frame += frame_qtd

        frame_label = np.zeros(frame_qtd)   # All frames here are normal.
        
        gt.append([video_path, frame_label])
        
    return gt

def test(dataloader, model_pt, model_ds, viz, max_sample_duration, list_, STRIDE_TEST, device, ten_crop, gt_path, OBJECTS_ALLOWED, N, T, EXIT_TOKEN, only_abnormal = False):

    dataloader = iter(dataloader)

    # Receber isso por parâmetro
    NUM_SAMPLE_FRAME = 15
    LABELS_PATH = os.path.join(DATASET_DIR, "videos/labels/test.txt")

    labels = getLabels(LABELS_PATH, list_) # 2d matrix containing the frame-level frame (columns) for each video (lines)

    gt = []
    scores = []
    temporal_graph = temporalGraph.TemporalGraph(device, T, OBJECTS_ALLOWED, N, STRIDE_TEST)    
    with torch.no_grad():
        model_pt.eval()
        model_ds.eval()    

        acc = 0
        for video_index, video in enumerate(labels):    # For each video

            # Adjust the labels to the truncated frame due computation capability militation
            truncated_frame_qtd = int((max_sample_duration) * NUM_SAMPLE_FRAME)    # The video has max this num of frame
            if len(video[1]) > truncated_frame_qtd:
                video[1] = video[1][0:truncated_frame_qtd]                  # If needed, truncate it

            qtdFrame = len(video[1])
            window = 0
            png_conter = 0

            # Initially lets discart the first NUM_SAMPLE * T Frames, it is the first sample
            input= next(dataloader)

            for i, j in enumerate(video[1]):     # For each label in this video

                acc+= 1

                # The first NUM_SAMPLE_FRAME * T frames will be discarted, because the incomplete window
                if i < (NUM_SAMPLE_FRAME * T):
                    continue

                if window == NUM_SAMPLE_FRAME or i == qtdFrame:
                    window = 0

                if window != 0:
                    scores.append(score)  
                    gt.append(j)                       
                    window += 1
                    continue

                input= next(dataloader)
                input = np.squeeze(input)
                png_conter += 1

                # We cannot accept to exist more .png than frames. If is the case, some MERDA occuried
                while input[2].cpu().flatten() != video_index:
                    print("Há mais png do que frames")
                    print("input: ")
                    print(input[3])
                    print("video index")
                    print(video_index)
                    #input = next(dataloader)
                    print("DEU MERDA. NÃO ERA PRA EXISTIR MAIS PNG DO QUE LABELS")
                    print("Qtd no gt: ")
                    print(len(gt))
                    print("qtd no scores: ")
                    print(len(scores))
                    exit()

                # Infs used to get inference result in the buffer, to preserve computing time
                folder_index = input[2]
                sample_index = input[3]


                frames = torch.squeeze(input[0])

                cache_folder = "cache_ds_task/test/"
                data_path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()[0]), str(sample_index.cpu().numpy()[0])+"_data.npy")
                has_cache = False
                if os.path.exists(data_path):
                    has_cache = True
                    dataloader.has_cache = True
                else:
                    has_cache = False
                    dataloader.has_cache = False

                if not has_cache:
                    adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(frames, folder_index, sample_index)

                    SIMILARITY_THRESHOLD = 0.65#0.73
                    reference_frame = 0
                    obj_predicted = 0
                    graph = utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, SIMILARITY_THRESHOLD, T, N)


                    data, object_path = utils.calculeTarget(graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, device, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

                    path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()[0]))
                    os.makedirs(path, exist_ok=True)

                    np.save(data_path, data)

                else:

                    data = np.load(data_path, allow_pickle=True)

                if data == -1:
                    # TODO: WE DO NOT CAN CONSIDER AS NORMAL JUST BECAUSE THERE IS NO OBJECTS IN THE FIRST FRAME.
                    # IF WE HAVE A BIGGER WINDOW, WE NEED CONSIDER THE OTHER FRAMES
                    score = 0.0

                else:

                    input, target = data 

                    # If the ins't any object on scene, we consider as NORMAL
                    if input is -1:
                        score = 0.0
                    else:
                        score = model_ds(model_pt(input))
             
                        score = score.data.cpu().numpy().flatten()[0]     
                    
                scores.append(score)  
                gt.append(j)   
                window += 1
                
            window = 0      

    
    #gt = list(gt)

    fpr, tpr, threshold = roc_curve(gt, scores)

    if only_abnormal:            
        np.save('fpr_graph_only_abnormal.npy', fpr)
        np.save('tpr_graph_only_abnormal.npy', tpr)
    else:
        np.save('fpr_graph.npy', fpr)
        np.save('tpr_graph.npy', tpr)

    rec_auc = auc(fpr, tpr)

    print('auc : ' + str(rec_auc))

    best_threshold = threshold[np.argmax(tpr - fpr)]
    print("Best threshold: ")
    print(best_threshold)

    precision, recall, th = precision_recall_curve(list(gt), scores)
    pr_auc = auc(recall, precision)
    np.save('precision.npy', precision)
    np.save('recall.npy', recall)
    viz.plot_lines('pr_auc', pr_auc)
    viz.plot_lines('auc', rec_auc)
    viz.lines('scores', scores)
    viz.lines('roc', tpr, fpr)
    return rec_auc



if __name__ == '__main__':

    gpu_id = 0
    args = {"gt": "list/gt-camnuvem.npy", "segment_size": 16}
    videos_pkl_test = "./arquivos/camnuvem-i3d-test-10crop.list"
    hdf5_path = "./arquivos/data_test.h5" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = 2048
    ten_crop = True

    model = HOE_model(nfeat=features, nclass=1, ten_crop=ten_crop)
    if gpu_id != -1:
        model = model.cuda(gpu_id)    

    test_loader = DataLoader(dataset_h5_test(videos_pkl_test, hdf5_path, ten_crop), pin_memory=False)

    auc = test(test_loader, model, args, device, ten_crop)                                  