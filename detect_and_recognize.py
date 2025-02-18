import cv2
from ultralytics import YOLO
import torch
import torch.backends.cudnn as cudnn
from utils import CTCLabelConverter, AttnLabelConverter
from model import Model
import argparse
from PIL import Image
from torchvision import transforms
import time
import numpy as np

# Inisialisasi YOLOv8 Model untuk deteksi plat nomor
yolo_model_path = '/home/bagus/Proposal/ultralytics/runs/detect/train14/weights/best.pt'
model = YOLO(yolo_model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_ocr_model(opt):
    """Inisialisasi model OCR dengan TPS-ResNet-BiLSTM-Attn"""
    converter = CTCLabelConverter(opt.character) if 'CTC' in opt.Prediction else AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    ocr_model = Model(opt).to(device)
    ocr_model = torch.nn.DataParallel(ocr_model)
    ocr_model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    print('OCR model loaded from:', opt.saved_model)

    return ocr_model, converter

def recognize_characters(model, converter, image, opt):
    """Proses OCR pada gambar plat nomor."""
    image = Image.fromarray(image).convert('L')

    transform = transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgW), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transformed_image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
        preds = model(transformed_image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        pred_str = converter.decode(preds_index, torch.IntTensor([opt.batch_max_length]))
        pred = pred_str[0].split('[s]')[0] if 'Attn' in opt.Prediction else pred_str[0]
    return pred.upper()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', default='saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_norm_ED.pth')
    parser.add_argument('--batch_max_length', type=int, default=25)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--Transformation', type=str, default='TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn')
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=256)

    opt = parser.parse_args()

    cudnn.benchmark = True
    opt.num_gpu = torch.cuda.device_count()

    ocr_model, converter = initialize_ocr_model(opt)

    # Buka video file
    video_path = '/home/bagus/Proposal/video_plat6.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi plat nomor dengan YOLO
        results = model(frame, imgsz=640, verbose=False)

        # Lakukan OCR untuk setiap plat yang terdeteksi
        for result in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, result)
            cropped_plate = frame[y1:y2, x1:x2]

            # OCR pada plat nomor
            if cropped_plate.size > 0:
                pred_text = recognize_characters(ocr_model, converter, cropped_plate, opt)
                print(f'Predicted Plate: {pred_text}')

                # Tampilkan teks hasil OCR pada frame
                cv2.putText(frame, pred_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        # Hitung FPS sistem
        end_time = time.time()
        elapsed_time = end_time - start_time
        system_fps = 1 / elapsed_time if elapsed_time > 0 else 0

        # Tampilkan FPS pada frame
        cv2.putText(frame, f"FPS: {system_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        # Tampilkan hasil di layar
        cv2.imshow('YOLOv8 + OCR Inference', frame)

        # Kontrol jalannya video
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()