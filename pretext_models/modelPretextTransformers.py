import torch
import torch.nn as nn
import torch.nn.init as torch_init

from transformers import ViTFeatureExtractor



# TODO:
# Weight init
class modelPretextTransformers(nn.Module):
    def __init__(self, num_feat_in, num_feat_out):
        super(modelPretextTransformers, self).__init__()	


        # import model
        model_id = 'google/vit-base-patch16-224-in21k'
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            model_id
        )

    def forward(self, inputs):	# Input é o ROI Align da Resnet. Não está limitado a 255. Range real é: [0, inf]

        example = self.feature_extractor(
            inputs,
            return_tensors='pt'
        )
        print(example)
        print(example.shape)
        


        return example['pixel_values']
