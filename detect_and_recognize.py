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
        # Ubah ke grayscale
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

            # Hitung confidence rata-rata dengan softmax
            probs = torch.nn.functional.softmax(preds, dim=2)
            max_probs, preds_index = probs.max(2)
            avg_confidence = max_probs.mean().item()

            pred_str = converter.decode(preds_index, torch.IntTensor([opt.batch_max_length]))
            pred = pred_str[0].split('[s]')[0] if 'Attn' in opt.Prediction else pred_str[0]

        # Hanya tampilkan jika confidence ≥ 90%
        if avg_confidence >= 0.9:
            return pred.upper(), avg_confidence
        else:
            return None, avg_confidence

    except Exception as e:
        print(f'[ERROR] OCR failed: {e}')
        return None, 0

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

    # Baca frame rate asli video
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps_video) if fps_video > 0 else 33

    # Penghitungan FPS yang realistis
    fps_timer = time.time()
    displayed_frames = 0
    fps_display = 0

    frame_count = 0
    last_predicted_plate = ""
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            displayed_frames += 1

            # Setiap 5 frame, lakukan deteksi dan OCR jika y1 > 340
            if frame_count % 5 == 0:
                results = yolo_model(frame, imgsz=640, verbose=False)

                if results and len(results) > 0 and len(results[0].boxes.xyxy) > 0:
                    box = results[0].boxes.xyxy[0]
                    x1, y1, x2, y2 = map(int, box)

                    # Hanya deteksi jika y1 > 340
                    if y1 > 340:
                        cropped_plate = frame[y1:y2, x1:x2]

                        if cropped_plate.size > 0:
                            pred_text, conf = recognize_characters(ocr_model, converter, cropped_plate, opt)
                            if pred_text:
                                print(f'[INFO] Predicted Plate: {pred_text} (Confidence: {conf:.2%})')
                                last_predicted_plate = f"{pred_text} ({conf:.2%})"

                                # Tampilkan bounding box dan teks
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, last_predicted_plate, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                # Pause setelah prediksi berhasil
                                paused = True

        # Tampilkan prediksi plat di pojok kanan atas
        if last_predicted_plate:
            text = f"Plate: {last_predicted_plate}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            x = frame.shape[1] - text_width - 10
            y = text_height + 10
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        # Hitung FPS setiap detik (hanya saat tidak pause)
        if not paused:
            current_time = time.time()
            if current_time - fps_timer >= 1.0:
                fps_display = displayed_frames / (current_time - fps_timer)
                displayed_frames = 0
                fps_timer = current_time

        # Tampilkan FPS di layar
        cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Tampilkan frame
        cv2.imshow('YOLOv8 + OCR Inference', frame)

        # Kontrol pause dan unpause
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):  # Keluar dengan 'q'
            break
        elif key == ord(' '):  # Unpause dengan Space
            paused = False

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai.")

if __name__ == '__main__':
    main()