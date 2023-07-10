import visdom
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
import PIL
from definitions import ROOT_DIR, FRAMES_DIR, DATASET_DIR
import os

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)


# Run in the temporal graph to localize the 'object predicted' through the all object detected in the self.T frames
def targetFind(object_predicted, adjacency_matrix, frame, SIMILARITY_THRESHOLD, T):

    object_path = np.empty(T)
    object_path.fill(-1)

    next_object = object_predicted

    object_path[0] = int(next_object)

    for i in range(len(adjacency_matrix)):

        # if between any frame there was no deteced object, then the object exits the scene
        # So we have to map the NN to EXIT_TOKEN
        if len(adjacency_matrix[i]) == 0:
            return -1, object_path
        obj = torch.argmax(adjacency_matrix[i], dim=1)

        # If the similarity of two objects in two consecutive frames are less than SIMILARITY_THRESHOLD,
        # we consider they aren't the same. 
        most_probably_next_object_index = obj[next_object]

        if adjacency_matrix[i][next_object][most_probably_next_object_index] < SIMILARITY_THRESHOLD:
            return -1, object_path

        next_object = most_probably_next_object_index
        object_path[i+1] = int(next_object)

    
    # Given a object in the first frame, self.T frames after this object was detected in the 'next_object' index of the object list
    return next_object, object_path

# Given a 'obj_predicted' in the 'reference_frame', calcule it's path though the frame window
def calculeTarget_deprecated(adj_mat, bbox_fea_list, box_list, reference_frame, obj_predicted, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N):

    target_index, object_path = targetFind(obj_predicted, adj_mat, reference_frame, SIMILARITY_THRESHOLD, T)

    # If object exits the scene, lets map the NN to EXIT_TOKEN
    if target_index == -1:
        target = torch.zeros(EXIT_TOKEN)
    else:
        object_in_the_last_frame = bbox_fea_list[-1][1][target_index]        # Last element, objects from last image, object num. 'obj_predicted'
        object_in_the_last_frame_box = box_list[-1][1][target_index]        # a 4 element list

        # The output is the x1, y1, x2, y2 object coordinates concat with object feature
        target = np.append(object_in_the_last_frame_box, object_in_the_last_frame)
        target = torch.from_numpy(target).to(DEVICE)

    # The input will be N-1 objects
    input_boxes = []
    input_features = []
        
    num_obj = len(bbox_fea_list[0][0])
    for i in range(N):
        input_boxes.append(box_list[reference_frame][0][i])
        input_features.append(bbox_fea_list[reference_frame][0][i])

    input_boxes = np.stack(input_boxes, axis=0)
    input_features = np.stack(input_features, axis=0)

    input = np.append(input_boxes, input_features)
    """
        # The input is the (x1, y1, x2, y2) * N-1 object coordinates concat with (object feature) * N-1
        input = bbox_fea_list[reference_frame][0][i]    # A feature
        input = np.append(box_list[reference_frame][0][obj_predicted], input)        
    """
    input = torch.from_numpy(input).to(DEVICE)

    return [input, target], object_path    

def calculeTarget(graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N):

    # First we have to calculate the path of first object, that will be our label.
    object_path = []

    # First frame without objects
    if len(graph[0]) == 0:
        return -1, -1

    object_path = [graph[i][0] for i in range(reference_frame, T)]

    if len(object_path) < T-1:
        print("Nã era para o object_path ser menor que T")
        exit()

    if object_path[-1] == -1:
        target = torch.zeros(EXIT_TOKEN)
    else:
        target_index = object_path[-1]

        object_in_the_last_frame = bbox_fea_list[-1][1][target_index]        # Last element, objects from last image, object num. 'obj_predicted'
        object_in_the_last_frame_box = box_list[-1][1][target_index]         # a 4 element list

        # The output is the x1, y1, x2, y2 object coordinates concat with object feature
        target = np.append(object_in_the_last_frame_box, object_in_the_last_frame).astype('float32')
        target = torch.from_numpy(target).to(DEVICE)


    # Now we have to calculate the input. The input will be N objects in each frame. 
    input_boxes = []
    input_features = []

    box_shape = len(box_list[0][0][0])
    fea_shape = len(bbox_fea_list[0][0][0])

    num_obj = len(bbox_fea_list[0][0])
    for i in range(reference_frame, T-1):

        # Initially a list of all objects in graph, includes several -1
        top_n = graph[i]

        # Remove all -1
        list2 = []
        for j in top_n:
            if j != -1:
                list2.append(j)
        top_n = list2

        top_n = top_n[0:N]

        for obj_index in top_n:
            input_boxes.append(box_list[i][0][obj_index])
            input_features.append(bbox_fea_list[i][0][obj_index])

    
        # ADD a token to explain to the NN that does not have more objects.
        cont = len(top_n)
        NO_ENTRY_TOKEN_boxes = [1.0 for i in range(box_shape)]
        NO_ENTRY_TOKEN_features = [1.0 for i in range(fea_shape)]

        while(cont < N):
            input_boxes.append(NO_ENTRY_TOKEN_boxes)
            input_features.append(NO_ENTRY_TOKEN_features)
            cont += 1
    
    input_boxes = np.stack(input_boxes, axis=0).astype('float32')
    input_features = np.stack(input_features, axis=0).astype('float32')

    input = np.append(input_boxes, input_features).astype('float32')
    """
     The input is the (x1, y1, x2, y2) * N-1 object coordinates concat with (object feature) * N-1
    """
    input = torch.from_numpy(input).to(DEVICE)                  

    # Return the input, the target and the object_path of reference object
    return [input, target], object_path

# Verify if there is N objects with high accuracy. Avoid to add spurius objects. 
# If a new object is in 'objects_already_tracked', does not include it as a new object
def add_objects(frame, T, N, score_list, objects_already_tracked):
    SCORE_THRESHOLD_NEW_OBJECT = 0.7

    list_ = []

    index = 0
    if frame == T-1:
        frame -= 1
        index = 1

    detected_objects_num = len(score_list[frame][index])

    #n_ = N
    #if detected_objects_num < N:
    #    n_ = detected_objects_num

    cont = 0
    for i in range(detected_objects_num):
        if score_list[frame][index][i] > SCORE_THRESHOLD_NEW_OBJECT:    
            if i not in objects_already_tracked:
                list_.append(i)
                cont += 1
        if cont == N:
            break            

    return list_

# Calcule the object path of all objects detected in every windows frame. 
def calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, SIMILARITY_THRESHOLD, T, N):

    graph = []  # 3d adjacency matryx, objects through time
    next_object = []

    graph.append([])
    for j in range(len(bbox_fea_list)):
    #for j in range(T-1):    # for each frame

        top_n = add_objects(j, T, N, score_list, next_object)
        graph[j].extend(top_n)
        next_object = []

        for i in graph[j]:  # For each object. Initially, 0 to N

            if i == -1:
                next_object.append(-1)
                continue

            target_index, _ = targetFind(i, [adj_mat[j]], reference_frame, SIMILARITY_THRESHOLD, T)
            next_object.append(int(target_index))

        graph.append(next_object)

    # Verify new objects in the last frame
    top_n = add_objects(j+1, T, N, score_list, next_object)
    graph[j+1].extend(top_n)  

    return graph

def fileLines2List(file):
    # Using readlines()
    file1 = open(file, 'r')
    Lines = file1.readlines()
      
    count = 0
    list_ = []
    # Strips the newline character
    for line in Lines:
        count += 1
        list_.append(line.strip()) 

    return list_


# pred2 = (boxes2, scores2, labels2, bbox_fea_vec2) idem for pred1
def print_image(input, bbox_list, object_path, index):

    str_labels = np.asarray(fileLines2List("../files/coco_labels.txt"))

    cont = 0

    print("Object path: ")
    print(object_path)
    for i in range(len(object_path)):

        image = input[i]
        image = image.cpu()

        object_interest = int(object_path[i])

        # If the object exits of scene, we will not print then
        if object_interest == -1:
            return

        # if we have 4 images, we have 3 bbox_list (each in th middle of each image pair),
        # containing the objects of the anterior (index 0) and posterior (index 1).
        # Yes, there are oject information duplicated
        if i == len(input)-1:
            box = bbox_list[i-1][1][object_interest]
        else:
            box = bbox_list[i][0][object_interest]

        boxes = np.asarray([box])

        image_tensor = torch.from_numpy(np.array(image))
        image_tensor = torch.moveaxis(image_tensor, 2, 0)
        boxes = torch.from_numpy(boxes)

        labels = ['1']

        print(boxes)
        img_com_bb = draw_bounding_boxes(image_tensor, boxes, labels)
        img_com_bb = torch.moveaxis(img_com_bb, 0, 2)
        img_com_bb = img_com_bb.numpy()
        print("Chegou aki")
        PIL.Image.fromarray(img_com_bb).convert("RGB").save("imagens/amostra-"+str(index)+"-"+str(cont)+".png")

        cont += 1
    

def img2bbox(input, temporal_graph, N_DOWNSTRAM):

    oi = input[0]


    batch_size = oi.shape[0]
    window_frame = oi.shape[1]


    res = []
    box_list = []
    for i in range(batch_size):
        batch = oi[i]
        for j in range(window_frame):
            img = oi[i,j]
            prediction = temporal_graph.inference(img)
            boxes, scores, labels, bbox_fea_vec  = temporal_graph.filterLowScores(prediction)

            if bbox_fea_vec.shape[0] < N_DOWNSTRAM:
                return -1, -1

            res.append(bbox_fea_vec[0:N_DOWNSTRAM])
            box_list.append(boxes[0:N_DOWNSTRAM])


    
    res = np.stack(res, axis=0)
    box_list = np.stack(box_list, axis=0)
    box_list = torch.from_numpy(box_list)
    box_list = box_list.view(oi.shape[0], oi.shape[1], N_DOWNSTRAM, 4)
    
    res = torch.from_numpy(res)
    res = res.view(oi.shape[0], oi.shape[1], N_DOWNSTRAM, bbox_fea_vec.shape[1])

    return res, box_list

def calculeObjectPath(graph, frame, obj_index):

    object_path = []

    print("calculando o path do objeto "+str(obj_index))
    print(graph[frame:])
    for f in graph[frame:]:     # For each frame
        print(f[obj_index])
        obj = f[obj_index]

        if obj == -1:
            return

        object_path.append(obj)
        
    return object_path


def batch_processing(input_abnormal, input_normal, temporal_graph_normal, temporal_graph_abnormal, normal_loader, abnormal_loader, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N):

    # In the end of the dataset, the dataloader can supply samples with differents sizes in the shape[0]. So, m
    # We need work with the lower one
    bath_len = min(input_abnormal[0].shape[0], input_normal[0].shape[0])

    batch_list = []
    for i in range(bath_len):

        
        input_normal_folder_index = input_normal[2][i]
        input_normal_sample_index = input_normal[3][i]
        input_normal_ = input_normal[0][i]

        input_abnormal_folder_index = input_abnormal[2][i]
        input_abnormal_sample_index = input_abnormal[3][i]
        input_abnormal_ = input_abnormal[0][i]

        cache_folder = "cache_ds_task/training/"
        data_nor_path = os.path.join(FRAMES_DIR, cache_folder, str(input_normal_folder_index.cpu().numpy()), str(input_normal_sample_index.cpu().numpy())+"_data_nor.npy")
        data_abn_path = os.path.join(FRAMES_DIR, cache_folder, str(input_normal_folder_index.cpu().numpy()), str(input_normal_sample_index.cpu().numpy())+"_data_abn.npy")
        has_cache = False
        if os.path.exists(data_nor_path) and os.path.exists(data_abn_path):
            has_cache = True
            normal_loader.has_cache = True
            abnormal_loader.has_cache = True
        else:
            has_cache = False

        if not has_cache:
            adj_mat_nor, bbox_fea_list_nor, box_list_nor, score_list_nor = temporal_graph_normal.frames2temporalGraph(input_normal_, input_normal_folder_index, input_normal_sample_index)
            adj_mat_abn, bbox_fea_list_abn, box_list_abn, score_list_abn = temporal_graph_abnormal.frames2temporalGraph(input_abnormal_, input_abnormal_folder_index, input_abnormal_sample_index)

            SIMILARITY_THRESHOLD = 0.65#0.73
            reference_frame = 0
            obj_predicted = 0
            graph_nor = calculeTargetAll(adj_mat_nor, bbox_fea_list_nor, box_list_nor, score_list_nor, reference_frame, SIMILARITY_THRESHOLD, T, N)
            graph_abn = calculeTargetAll(adj_mat_abn, bbox_fea_list_abn, box_list_abn, score_list_abn, reference_frame, SIMILARITY_THRESHOLD, T, N)

            data_nor, object_path_nor = calculeTarget(graph_nor, score_list_nor, bbox_fea_list_nor, box_list_nor, reference_frame, obj_predicted, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)
            data_abn, object_path_abn = calculeTarget(graph_abn, score_list_abn, bbox_fea_list_abn, box_list_abn, reference_frame, obj_predicted, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

            path = os.path.join(FRAMES_DIR, cache_folder, str(input_normal_folder_index.cpu().numpy()))
            os.makedirs(path, exist_ok=True)

            np.save(data_nor_path, data_nor)
            np.save(data_abn_path, data_abn)

        else:
            data_nor = np.load(data_nor_path, allow_pickle=True)
            data_abn = np.load(data_abn_path, allow_pickle=True)
            #adj_mat, bbox_fea_list, box_list, score_list = np.load(fea_path, allow_pickle=True).tolist()
            #graph_nor = np.load(graph_path, allow_pickle=True)        

        # First frame without objects
        if data_nor == -1 or data_abn == -1:
            continue

        input_normal_, target_normal = data_nor 
        input_abnormal_, target_abnormal = data_abn 

        batch_list.append([input_abnormal_, input_normal_])

    return batch_list