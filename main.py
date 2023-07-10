## Exemplo de: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Seems we do not need normalization for pytorch built-in models: https://github.com/pytorch/vision/blob/d2c763e14efe57e4bf3ebf916ec243ce8ce3315c/torchvision/models/detection/faster_rcnn.py#L227

# Explicação sobre Roi Polling e Roi Aling: 
# https://erdem.pl/pages/work

# Talvez um caminho de como acessar as features no RoiAlign:
#https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection



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



OBJECT_DETECTION_THESHOLD = 0.55


# Exemplo de como trabalhar com hook: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def loadImage():
    #image = PIL.Image.open("zidane.jpeg").convert('RGB').convert('RGB')
    image = cv2.imread("zidane.jpeg")
    

    return image

def prediction2image(prediction, original_image):
    scores = prediction[0]['scores'].cpu().detach().numpy()
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()

    boxes, scores, labels = filterLowScores(boxes, scores, labels)
    boxes = torch.from_numpy(boxes) 
    scores = torch.from_numpy(scores)  
    labels = labels 


    image_tensor = torch.from_numpy(np.array(original_image))
    image_tensor = torch.moveaxis(image_tensor, 2, 0)


    img_com_bb = draw_bounding_boxes(image_tensor, boxes, labels)
    img_com_bb = torch.moveaxis(img_com_bb, 0, 2)


    img_com_bb = img_com_bb.numpy()


    PIL.Image.fromarray(img_com_bb).convert("RGB").save("art.png")


def filterLowScores(boxes, scores, labels):

    new_boxes = []
    new_scores = []
    new_labels = []
    for i in range(boxes.shape[0]):
        if scores[i] > OBJECT_DETECTION_THESHOLD:
            new_boxes.append(boxes[i])
            new_scores.append(scores[i])
            new_labels.append(str(labels[i]))

    return np.asarray(new_boxes), np.asarray(new_scores), np.asarray(new_labels)


def run():

    labels = ""
    num_classes = 91
    MODEL = "frcnn-mobilenet"

    # set the device we will be using to run the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the list of categories in the COCO dataset and then generate a
    # set of bounding box colors for each class
    #CLASSES = pickle.loads(open(labels, "rb").read())
    COLORS = np.random.uniform(0, 255, size=(num_classes, 3))



    # initialize a dictionary containing model name and its corresponding 
    # torchvision function call
    MODELS = {
        "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
        "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        "retinanet": detection.retinanet_resnet50_fpn
    }
    # load the model and set it to evaluation mode
    #model = 
    #model.eval()
    #print(model)

    




    # load the image from disk
    #image = cv2.imread(args["image"])

    image = loadImage() 
    print(image)
    orig = image.copy()
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
    print(image.shape)
    print(torch.amax(image))
    print(torch.amin(image))

    start = time.time()
    #detections = model(image)

    model = MODELS[MODEL](pretrained=True, progress=True,num_classes=num_classes, pretrained_backbone=True).cuda()
    model.eval()
    print(model)
    #print(model.roi_heads.box_roi_pool)
    model.roi_heads.box_predictor.cls_score.register_forward_hook(get_activation('cls_score'))
    model.roi_heads.box_predictor.bbox_pred.register_forward_hook(get_activation('bbox_pred'))
    model(image)
    print("mels dels")
    print(activation['cls_score'].shape)

    scores = activation['cls_score']
    bbox = activation['bbox_pred']
    cont= 0 
    for cat_scores in range(scores.shape[0]):
        print("Objeto " + str(cont))
        print("Index da categoria com score mais alto: ")
        indice = torch.argmax(scores[cat_scores])
        print(indice)
        print("Valor: ")
        print(scores[cat_scores][indice])
        print("\n")

        cont += 1

    exit()



    print("O maior score é: ")
    print(torch.amax(activation['cls_score']))
    print(activation['cls_score'][0])
    print(torch.argmax(activation['cls_score'], 1))
    print(activation['bbox_pred'].shape)
    #result = model(image)
    #print(result)
    exit()

    #resnet_features = FeatureExtractor(.to(DEVICE), layers=["roi_heads", "rpn", ""])
    resnet_features.eval()

    print(resnet_features.FasterRCNN)
    exit()

    print(resnet_features(image))
    #features = resnet_features(dummy_input)
    exit()

    print("\n\nTempo decorrido: " + str(time.time() - start) + "\n\n")

    print(detections)
    prediction2image(detections, orig)


    exit()


    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN (Region Proposan Network) generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    # Alguma explicação: https://www.alegion.com/faster-r-cnn#:~:text=Anchor%20boxes%20are%20some%20of,object%20locations%20for%20the%20RPN.
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    print(anchor_generator)

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    model.eval()
                         


    #print(sample)
    #print(np.amax(sample))
    #print(np.amin(sample))    


    # Normalization for mobile_net_v2 here: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    preprocess = transforms.Compose([
        
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(sample)
    #print(input_tensor)
    #print(torch.amax(input_tensor))
    #print(torch.amin(input_tensor))

    #exit()

    input_batch = input_tensor.unsqueeze(0)
    print(input_batch)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')



    #sample = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

    prediction = model(input_batch)

    prediction2image(prediction)

    


if __name__ == '__main__':
    run()
