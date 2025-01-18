import cv2
from ultralytics import YOLO
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate
from model import Model
import argparse
from PIL import Image  # Import PIL.Image
from torchvision import transforms
import os  # Import os module to clear console
import time  # Import time module to measure latency


# Inisialisasi YOLOv8 Model untuk deteksi plat nomor
yolo_model_path = '/home/bagus/Proposal/ultralytics/runs/detect/train9/weights/best.pt'
model = YOLO(yolo_model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_ocr_model(opt):
    """Inisialisasi model OCR dengan TPS-ResNet-BiLSTM-Attn"""
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    ocr_model = Model(opt)
    ocr_model = torch.nn.DataParallel(ocr_model).to(device)

    # Load model
    print('loading pretrained model from %s' % opt.saved_model)
    ocr_model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    return ocr_model, converter

def recognize_characters(model, converter, image, opt):
    """Recognize characters in the cropped license plate region."""
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(image)

    # Convert image to grayscale (1 channel)
    image = image.convert('L')
    
    # Define transformations manually as in AlignCollate
    transform = transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgW), interpolation=2),  # Resize to the expected input size
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])
    
    # Apply transformation to the cropped image
    transformed_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Prediction
    model.eval()
    with torch.no_grad():
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
        
        # Measure latency
        start_time = time.time()  # Record start time
        if 'CTC' in opt.Prediction:
            preds = model(transformed_image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)])
            _, preds_index = preds.max(2)
            pred_str = converter.decode(preds_index, preds_size)
        else:
            preds = model(transformed_image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            pred_str = converter.decode(preds_index, length_for_pred)
        end_time = time.time()  # Record end time

        # Calculate latency
        latency = end_time - start_time
        print(f"Latency: {latency:.4f} seconds")

        # Process output and remove unnecessary characters
        pred = pred_str[0]
        if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]
        
        return pred.upper()  # Return the recognized text

def main():
    # Parsing argument yang digunakan pada demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='demo_image/', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_norm_ED.pth', help="path to saved_model to evaluation")
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # Inisialisasi model OCR
    ocr_model, converter = initialize_ocr_model(opt)

    # Buka video file
    # video_path = '/home/bagus/Proposal/video_plat5.mp4'
    video_path = '/home/bagus/Proposal/Footage/12 MEI 2023/10.17.10.1_IP Camera2_10.17.10.1_20230512153255_20230512160204_167816630.mp4'
    cap = cv2.VideoCapture(video_path)  # Ganti ke 0 untuk kamera

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame with custom imgsize
        results = model(frame, imgsz=640, verbose=False)  # Set the imgsize (e.g., 640), verbose=False untuk menghilangkan print dari YOLO

        # Process results
        for result in results[0].boxes.xyxy:  # xyxy format
            x1, y1, x2, y2 = map(int, result)  # Koordinat bounding box

            # Crop the ROI from the frame
            cropped_plate = frame[y1:y2, x1:x2]

            # Lakukan OCR pada ROI
            pred_text = recognize_characters(ocr_model, converter, cropped_plate, opt)
            print(f'Predicted Text: {pred_text}')

        # Display the frame with annotations
        annotated_frame = results[0].plot()  # Plot the results on the frame
        cv2.imshow('YOLOv8 + OCR Inference', annotated_frame)

        # Pause if 'p' is pressed, continue if 'c' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            print("Inference paused. Press 'c' to continue.")
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == ord('c'):
                    print("Continuing inference...")
                    break
        elif key == ord('q'):
            break  # Exit if 'q' is pressed

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
