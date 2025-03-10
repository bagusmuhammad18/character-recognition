import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}
        
        # Transformation
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, 
                I_size=(opt.imgH, opt.imgW), 
                I_r_size=(opt.imgH, opt.imgW), 
                I_channel_num=opt.input_channel
            )
        else:
            print('No Transformation module specified')
        
        # Feature Extraction
        if opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(1, opt.output_channel)  # 1 channel untuk grayscale
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        
        # Sequence Modeling
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            )
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output
        
        # Prediction
        if opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        # Preprocessing grayscale
        if input.size(1) == 3:  # Jika gambar RGB, konversi ke grayscale
            input = 0.2989 * input[:, 0:1, :, :] + 0.5870 * input[:, 1:2, :, :] + 0.1140 * input[:, 2:3, :, :]

        # Feature extraction stage
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        # Sequence modeling stage
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature

        # Prediction stage
        if self.stages['Pred'] == 'Attn':
            prediction = self.Prediction(
                contextual_feature.contiguous(), 
                text, 
                is_train, 
                batch_max_length=self.opt.batch_max_length
            )
        else:
            prediction = self.Prediction(contextual_feature.contiguous())
        
        return prediction
