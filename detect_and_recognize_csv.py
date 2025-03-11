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
import csv
import os

# Inisialisasi perangkat (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_ocr_model(opt):
    """Inisialisasi model OCR dengan TPS-ResNet-BiLSTM-Attn"""
    converter = CTCLabelConverter(opt.character) if 'CTC' in opt.Prediction else AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    ocr_model = Model(opt).to(device)
    ocr_model = torch.nn.DataParallel(ocr_model)
    ocr_model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    print(f'[INFO] OCR model loaded from: {opt.saved_model}')

    return ocr_model, converter

def recognize_characters(model, converter, image, opt):
    """Proses OCR pada gambar plat nomor dengan threshold confidence 90%."""
    try:
        image = Image.fromarray(image).convert('L')
        transform = transforms.Compose([
            transforms.Resize((opt.imgH, opt.imgW)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        transformed_image = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            preds = model(transformed_image, text_for_pred, is_train=False)

            probs = torch.nn.functional.softmax(preds, dim=2)
            max_probs, preds_index = probs.max(2)
            avg_confidence = max_probs.mean().item()

            pred_str = converter.decode(preds_index, torch.IntTensor([opt.batch_max_length]))
            pred = pred_str[0].split('[s]')[0] if 'Attn' in opt.Prediction else pred_str[0]

        if avg_confidence >= 0.9:
            return pred.upper(), avg_confidence
        else:
            return None, avg_confidence

    except Exception as e:
        print(f'[ERROR] OCR failed: {e}')
        return None, 0

def is_box_within_margins(box, frame_shape, margin=50):
    """Cek apakah bounding box memiliki jarak minimal dari ujung frame."""
    x1, y1, x2, y2 = map(int, box)
    height, width = frame_shape[:2]
    
    if (x1 < margin or y1 < margin or 
        x2 > width - margin or y2 > height - margin):
        return False
    return True

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
    parser.add_argument('--video_path', type=str, default='/home/bagus/Proposal/C002_resized.mp4')
    parser.add_argument('--yolo_model_path', type=str, default='/home/bagus/Proposal/ultralytics/runs/detect/train14/weights/best.pt')
    parser.add_argument('--margin', type=int, default=50, help='Minimum distance from frame edges')
    parser.add_argument('--csv_output', type=str, default='fps_results.csv', help='Path to output CSV file')

    opt = parser.parse_args()

    cudnn.benchmark = True
    opt.num_gpu = torch.cuda.device_count()

    # Inisialisasi model OCR
    ocr_model, converter = initialize_ocr_model(opt)

    # Inisialisasi model YOLO
    print(f'[INFO] Loading YOLO model from: {opt.yolo_model_path}')
    yolo_model = YOLO(opt.yolo_model_path).to(device)

    # Buka video file
    cap = cv2.VideoCapture(opt.video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    # Variabel untuk FPS Pipeline Penuh
    total_start_time = time.time()
    total_frames = 0

    # Variabel untuk FPS Deteksi + Pengenalan Karakter
    detection_recognition_times = []

    frame_count = 0
    last_predicted_plate = ""

    # Siapkan file CSV
    csv_file = open(opt.csv_output, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'FPS_Pipeline', 'FPS_Detection_Recognition'])

    while cap.isOpened():
        # Mulai pengukuran waktu pipeline penuh untuk frame ini
        frame_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        total_frames += 1

        # Setiap 10 frame, lakukan deteksi dan OCR
        if frame_count % 10 == 0:
            # Mulai pengukuran waktu deteksi + pengenalan karakter
            detection_start_time = time.time()

            results = yolo_model(frame, imgsz=640, verbose=False)

            if results and len(results) > 0 and len(results[0].boxes.xyxy) > 0:
                box = results[0].boxes.xyxy[0]
                x1, y1, x2, y2 = map(int, box)

                # Hanya deteksi jika y1 > 340 dan ada jarak minimal dari ujung frame
                if (y1 > 340 and is_box_within_margins(box, frame.shape, margin=opt.margin)):
                    cropped_plate = frame[y1:y2, x1:x2]

                    if cropped_plate.size > 0:
                        pred_text, conf = recognize_characters(ocr_model, converter, cropped_plate, opt)
                        if pred_text:
                            print(f'[INFO] Frame {frame_count}: Predicted Plate: {pred_text} (Confidence: {conf:.2%})')
                            last_predicted_plate = f"{pred_text} ({conf:.2%})"

                            # Tampilkan bounding box dan teks
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, last_predicted_plate, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Selesai pengukuran waktu deteksi + pengenalan karakter
            detection_end_time = time.time()
            detection_time = detection_end_time - detection_start_time
            if detection_time > 0:
                fps_detection_recognition = 1 / detection_time
                detection_recognition_times.append(fps_detection_recognition)
            else:
                fps_detection_recognition = 0

        # Tampilkan prediksi plat di bawah FPS
        if last_predicted_plate:
            text = f"Plate: {last_predicted_plate}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            x = 10
            y = 60
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        # Hitung FPS pipeline penuh untuk frame ini
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        fps_pipeline = 1 / frame_time if frame_time > 0 else 0

        # Tampilkan FPS pipeline di layar
        cv2.putText(frame, f"Pipeline FPS: {fps_pipeline:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Tulis FPS ke CSV
        fps_detection_recognition = detection_recognition_times[-1] if frame_count % 10 == 0 else 0
        csv_writer.writerow([frame_count, fps_pipeline, fps_detection_recognition])

        # Tampilkan frame
        cv2.imshow('YOLOv8 + OCR Inference', frame)

        # Keluar dengan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Hitung rata-rata FPS
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_fps_pipeline = total_frames / total_time if total_time > 0 else 0
    avg_fps_detection_recognition = sum(detection_recognition_times) / len(detection_recognition_times) if detection_recognition_times else 0

    # Tulis rata-rata ke CSV
    csv_writer.writerow(['Average', avg_fps_pipeline, avg_fps_detection_recognition])

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    print(f"[INFO] Program selesai.")
    print(f"[INFO] Average Pipeline FPS: {avg_fps_pipeline:.2f}")
    print(f"[INFO] Average Detection+Recognition FPS: {avg_fps_detection_recognition:.2f}")
    print(f"[INFO] FPS results saved to {opt.csv_output}")

if __name__ == '__main__':
    main()