## Exemplo de: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Seems we do not need normalization for pytorch built-in models: https://github.com/pytorch/vision/blob/d2c763e14efe57e4bf3ebf916ec243ce8ce3315c/torchvision/models/detection/faster_rcnn.py#L227
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import numpy as np
import PIL
from torchvision import transforms, datasets

from torchvision.utils import draw_bounding_boxes

OBJECT_DETECTION_THESHOLD = 0.55

def loadImage():
    image = PIL.Image.open("zidane.jpeg").convert('RGB').convert('RGB')
    

    return image

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
                         


    sample = loadImage()
    #sample = np.asarray(sample)
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

    scores = prediction[0]['scores'].cpu().detach().numpy()
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()

    boxes, scores, labels = filterLowScores(boxes, scores, labels)
    boxes = torch.from_numpy(boxes) 
    scores = torch.from_numpy(scores)  
    labels = labels 


    image_tensor = torch.from_numpy(np.array(sample))
    image_tensor = torch.moveaxis(image_tensor, 2, 0)


    img_com_bb = draw_bounding_boxes(image_tensor, boxes, labels)
    img_com_bb = torch.moveaxis(img_com_bb, 0, 2)


    img_com_bb = img_com_bb.numpy()


    PIL.Image.fromarray(img_com_bb).convert("RGB").save("art.png")



if __name__ == '__main__':
    run()
