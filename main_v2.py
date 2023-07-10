## Exemplo de: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Seems we do not need normalization for pytorch built-in models: https://github.com/pytorch/vision/blob/d2c763e14efe57e4bf3ebf916ec243ce8ce3315c/torchvision/models/detection/faster_rcnn.py#L227

# Explicação sobre Roi Polling e Roi Aling: 
# https://erdem.pl/pages/work

# Talvez um caminho de como acessar as features no RoiAlign:
#https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection

# Conferir esse feature extraction depois!
# https://github.com/pytorch/vision/pull/4302

import torch
import torchvision
from torchvision.models import detection
import numpy as np
import PIL
from torchvision import transforms, datasets

from torchvision.utils import draw_bounding_boxes
import pickle
import cv2
import time
import torch.nn as nn
from typing import Dict, Iterable, Callable
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, mobilenet_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps
from .dataset_pretext import DatasetPretext
import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.sections()
config.read('config.ini')
T = int(config['PARAMS']['T'])

OBJECT_DETECTION_THESHOLD = 0.55

num_classes = 91

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

def extractFrames():
    capture = cv2.VideoCapture('./video_teste/2.mp4')
    c=0
    frame_list = []
    cont = 0
    while capture.isOpened():
        r, f = capture.read()
        if r == False:
            break
        if c % 15 == 0:
            cv2.imwrite('./video_teste/kang'+str(c)+'.jpg',f)
            frame_list.append(f)
            cont += 1
            if cont > 90:
                break
        c+=1

    capture.release()
    cv2.destroyAllWindows()    

    return frame_list


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

# pred2 = (boxes2, scores2, labels2, bbox_fea_vec2) idem for pred1
def print_image(adjacency_matrix, image, pred, index_obj, index):
    

    str_labels = np.asarray(fileLines2List("coco_labels.txt"))
    
    #obj = pred[0]
    #obj = torch.argmax(adjacency_matrix, dim=1)
    #obj1 = obj[0]   # 0 -> obj[0]  índices em boxes

    #filterLowScores(boxes1, scores1, str_labels1, bbox_fea_vec1)        

    labels = pred[2][index_obj]
    #labels2 = pred2[2][obj1]
    str_labels = str_labels[labels-1]
    #str_labels1 = str_labels[labels2-1]

    boxes = np.asarray([pred[0][index_obj]])
    print(boxes)
    #exit()
    #boxes2 = pred2[0][obj1]

    image_tensor = torch.from_numpy(np.array(image))
    image_tensor = torch.moveaxis(image_tensor, 2, 0)

    #image_tensor2 = torch.from_numpy(np.array(image1))
    #image_tensor2 = torch.moveaxis(image_tensor2, 2, 0)    


    boxes = torch.from_numpy(boxes)

    labels = list(map(str, [labels]))
    print(labels)


    #boxes2 = torch.from_numpy(boxes2)
    #labels2 = list(map(str, labels2))    

    #print(labels2)
    #print(labels1.shape)
    #print(labels1.dtype)

    print(boxes)
    #exit()
    img_com_bb = draw_bounding_boxes(image_tensor, boxes, labels, font="Plane Crash.ttf")
    img_com_bb = torch.moveaxis(img_com_bb, 0, 2)

    #img_com_bb2 = draw_bounding_boxes(image_tensor2, boxes2, labels2, font="Plane Crash.ttf")
    #img_com_bb2 = torch.moveaxis(img_com_bb, 0, 2)    


    img_com_bb = img_com_bb.numpy()


    PIL.Image.fromarray(img_com_bb).convert("RGB").save("art"+str(index)+".png")
    #PIL.Image.fromarray(img_com_bb2).convert("RGB").save("art"+str(index+1)+".png")    



def prediction2image(prediction, original_image):
    scores = prediction[0]['scores'].cpu().detach().numpy()
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()
    print(labels)

    # read coco labels
    str_labels = np.asarray(fileLines2List("coco_labels.txt"))
    str_labels = str_labels[labels-1]


    # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP
    bbox_fea_vec = prediction[0]['bbox_fea_vec'].cpu().detach().numpy()


    boxes, scores, labels = filterLowScores(boxes, scores, str_labels)
    boxes = torch.from_numpy(boxes) 
    scores = torch.from_numpy(scores)  
    labels = labels 


    image_tensor = torch.from_numpy(np.array(original_image))
    image_tensor = torch.moveaxis(image_tensor, 2, 0)


    img_com_bb = draw_bounding_boxes(image_tensor, boxes, labels, font="Plane Crash.ttf")
    img_com_bb = torch.moveaxis(img_com_bb, 0, 2)


    img_com_bb = img_com_bb.numpy()


    PIL.Image.fromarray(img_com_bb).convert("RGB").save("art.png")


def filterLowScores(prediction):


    # read coco labels
    str_labels = np.asarray(fileLines2List("coco_labels.txt"))

    scores = prediction[0]['scores'].cpu().detach().numpy()
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()
    bbox_fea_vec = prediction[0]['bbox_fea_vec'].cpu().detach().numpy()     # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP
    #str_labels1 = str_labels[labels1-1]

    new_boxes = []
    new_scores = []
    new_labels = []
    new_bbox_fea_vec = []
    for i in range(boxes.shape[0]):
        if scores[i] > OBJECT_DETECTION_THESHOLD:
            new_boxes.append(boxes[i])
            new_scores.append(scores[i])
            new_labels.append(labels[i])
            new_bbox_fea_vec.append(bbox_fea_vec[i])

    return np.asarray(new_boxes), np.asarray(new_scores), np.asarray(new_labels), np.asarray(new_bbox_fea_vec)

# (boxes1, scores1, labels1, bbox_fea_vec1)
def make_temporal_graph(pred1, pred2):

    # read coco labels
    str_labels = np.asarray(fileLines2List("coco_labels.txt"))

    scores1 = pred1[1]          # pred1[0]['scores'].cpu().detach().numpy()
    boxes1 = pred1[0]           # pred1[0]['boxes'].cpu().detach().numpy()
    labels1 = pred1[2]          # pred1[0]['labels'].cpu().detach().numpy()
    bbox_fea_vec1 = pred1[3]     #pred1[0]['bbox_fea_vec'].cpu().detach().numpy()     # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP

    str_labels1 = str_labels[labels1-1]


    scores2 = pred1[1]          # pred2[0]['scores'].cpu().detach().numpy()
    boxes2 = pred1[0]           # pred2[0]['boxes'].cpu().detach().numpy()
    labels2 = pred1[2]          # pred2[0]['labels'].cpu().detach().numpy()     
    bbox_fea_vec2 = pred1[3]    # pred2[0]['bbox_fea_vec'].cpu().detach().numpy()     # TODO: Veiricar se é melhor pegar as features diretamente da ResNet, antes do MLP
    str_labels2 = str_labels[labels2-1]
    

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    mat = np.zeros((bbox_fea_vec1.shape[0], bbox_fea_vec2.shape[0]))

    #          vec2[0] vec2[1] vec2[2] ..
    #vec1[0]     x       y        z   
    #vec1[1]    ...
    #vec1[2]
    # ...

    bbox_fea_vec1 = torch.from_numpy(bbox_fea_vec1)
    bbox_fea_vec2 = torch.from_numpy(bbox_fea_vec2)
    vec1_ = torch.repeat_interleave(bbox_fea_vec1, bbox_fea_vec2.shape[0], dim=0)
    print(bbox_fea_vec1.shape)
    print(bbox_fea_vec2.shape)
    vec2_ = bbox_fea_vec2.repeat(bbox_fea_vec1.shape[0], 1)

    print(vec1_.shape)
    print(vec2_.shape)

    output = cos(vec1_, vec2_)

    # Yeeep, here we have a adjacency matrix
    output = output.view(bbox_fea_vec1.shape[0], bbox_fea_vec2.shape[0])
    print(bbox_fea_vec1.shape)
    print(bbox_fea_vec2.shape)

    return output






    exit()

    
    


    
    boxes = torch.from_numpy(boxes) 
    scores = torch.from_numpy(scores)  
    labels = labels 


def inference(image, model):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    
    # convert the image from BGR to RGB channel ordering and change the
    # image from channels last to channels first ordering
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    # send the input to the device and pass the it through the network to
    # get the detections and predictions
    image = image.to(DEVICE)
    #print(image.shape)
    #print(torch.amax(image))
    #print(torch.amin(image))

    #print(model)
    prediction = model(image)    

    return prediction

def run():

    labels = ""
    
    MODEL = "frcnn-mobilenet"

    # set the device we will be using to run the model
    
    # load the list of categories in the COCO dataset and then generate a
    # set of bounding box colors for each class
    #CLASSES = pickle.loads(open(labels, "rb").read())
    COLORS = np.random.uniform(0, 255, size=(num_classes, 3))


    model_urls = {
        'fasterrcnn_resnet50_fpn_coco':
            'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
            'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
        'fasterrcnn_mobilenet_v3_large_fpn_coco':
            'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
    }


    pretrained=True
    progress=True
    pretrained_backbone=True
    trainable_backbone_layers = None


    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    model = model.cuda()
    model.eval()

    #image = loadImage() 
    images = extractFrames()


    #image1 = inference(images[0], model)


    index_obj1 = 0
    cont = 0
    image1 = images[0]
    prediction1 = inference(image1, model)
    boxes1, scores1, labels1, bbox_fea_vec1  = filterLowScores(prediction1)      
    data1 = (boxes1, scores1, labels1, bbox_fea_vec1)    
    for i in range(1, len(images)-1):
        #image1 = images[i]
        image2 = images[i]
        
        prediction2 = inference(image2, model)

        
        boxes2, scores2, labels2, bbox_fea_vec2  = filterLowScores(prediction2)    


        
        data2 = (boxes2, scores2, labels2, bbox_fea_vec2)
        adjacency_matrix = make_temporal_graph(data1, data2)


        obj = torch.argmax(adjacency_matrix, dim=1)
        
        index_obj2 = obj[index_obj1]
        print_image(adjacency_matrix, image1, data1, index_obj1, cont)
        prediction1 = prediction2
        image1 = image2
        data1 = data2
        index_obj1 = index_obj2
        #exit()
        cont += 1


    exit()
    image = images[0]
    orig = image.copy()

    start = time.time()
    prediction = inference(image, model)
    end = time.time()
    print("Predição levou " + str(end-start) + " sec")
    
    # O QUU EU QUETO ESTÁ EM 
    #/home/denis/.local/lib/python3.6/site-packages/torchvision/models/detection/roi_heads.py
    # PROCURE POR 
    # box_features = self.box_head(box_features)

    
    prediction2image(prediction, orig)
    

if __name__ == '__main__':


    batch_size = 1      # Aumentar no final
    # each video is a folder number-nammed
    training_folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames/training/normal"
    train_nloader = DataLoader(DatasetPretext(T, training_folder),
                                   batch_size=1, shuffle=False,
                                   num_workers=0, pin_memory=False)    
    run()

