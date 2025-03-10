import torch
import torch.nn as nn
import torch.nn.functional as F  # Untuk operasi pooling
import matplotlib.pyplot as plt
import cv2  # Pastikan OpenCV terinstal: pip install opencv-python
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
        self.image_counter = 0  # Inisialisasi counter untuk menyimpan gambar thresholded

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, 
                I_size=(opt.imgH, opt.imgW), 
                I_r_size=(opt.imgH, opt.imgW), 
                I_channel_num=opt.input_channel
            )
        else:
            print('No Transformation module specified')

        """ Feature Extraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence Modeling """
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            )
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Preprocessing stage: Grayscale dan Binarisasi dengan Adaptive Thresholding berbasis Local Mean """
        # Jika input memiliki 3 channel (RGB), konversi menjadi grayscale
        if input.size(1) == 3:  # input shape: (batch_size, channels, height, width)
            processed = 0.2989 * input[:, 0:1, :, :] + 0.5870 * input[:, 1:2, :, :] + 0.1140 * input[:, 2:3, :, :]
        else:
            processed = input  # Jika sudah grayscale (1 channel), gunakan langsung

        # Binarisasi: Gunakan adaptive thresholding berbasis local mean
        kernel_size = 15  # Ukuran kernel untuk menghitung rata-rata lokal, bisa disesuaikan
        padding = kernel_size // 2  # Padding agar ukuran output tidak berubah
        local_mean = F.avg_pool2d(processed, kernel_size, stride=1, padding=padding)  # Hitung rata-rata lokal
        offset = 0.05  # Offset untuk menyesuaikan threshold, dapat disesuaikan
        processed = (processed > (local_mean - offset)).float()  # Terapkan thresholding, hasilnya 0 atau 1

        # Simpan hasil thresholding (hanya batch pertama untuk efisiensi)
        if self.image_counter < 100:  # Batasi penyimpanan maksimal 100 gambar
            img_to_save = processed[0].squeeze().cpu().numpy()  # Ambil gambar pertama dari batch
            plt.imsave(f'threshold/threshold_{self.image_counter:03d}.png', img_to_save, cmap='gray')
            self.image_counter += 1

        """ Operasi Morfologi (Dikomentari Sesuai Permintaan) """
        # dilated = F.max_pool2d(processed, kernel_size=3, stride=1, padding=1)  # Dilasi
        # eroded = 1 - F.max_pool2d(1 - processed, kernel_size=3, stride=1, padding=1)  # Erosi
        # opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)  # Opening
        # closed = 1 - F.max_pool2d(1 - dilated, kernel_size=3, stride=1, padding=1)  # Closing
        # processed = opened  # Misal: menggunakan hasil operasi opening

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(processed)  # Gunakan input yang sudah diproses
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(), 
                text, 
                is_train, 
                batch_max_length=self.opt.batch_max_length
            )

        return prediction