import torch
import torch.nn as nn
import torch.nn.init as torch_init
import definitions
import os
import sys



YOLOV5_ROOT = os.path.join(definitions.ROOT_DIR, '../../../')
print(YOLOV5_ROOT)
if str(YOLOV5_ROOT) not in sys.path:
    print("porra")
    sys.path.append(str(YOLOV5_ROOT))  # add ROOT to PATH
    sys.path.append(str(YOLOV5_ROOT)+'/yolov5')  # add ROOT to PATH

print(sys.path)
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)


imgsz=(640, 640)
agnostic_nms = False
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
classes=None
weights = os.path.join(YOLOV5_ROOT, 'yolov5s.pt')
dnn=False
data = os.path.join(YOLOV5_ROOT, 'data/coco128.yaml')
half = False
bs = 1
# Load model
device = '0'
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

class ModelPretextYolov5(nn.Module):
    def __init__(self, num_feat_in, num_feat_out):
        print("Chegou aiki")
        exit()
        pass