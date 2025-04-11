import os
import time
import string
import argparse
import re
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
import Levenshtein as lev  # Tambahkan library Levenshtein untuk analisis kesalahan

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fungsi untuk menghitung kesalahan substitusi, insersi, dan penghapusan
def compute_error_types(gt, pred):
    ops = lev.editops(gt, pred)
    substitution_count = len([op for op in ops if op[0] == 'replace'])
    insertion_count = len([op for op in ops if op[0] == 'insert'])
    deletion_count = len([op for op in ops if op[0] == 'delete'])
    return substitution_count, insertion_count, deletion_count

# Fungsi untuk menyimpan gambar dengan anotasi
def save_image_with_annotation(image, gt, pred, error_type, opt, idx):
    output_dir = f'./result/{opt.exp_name}/{error_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Konversi tensor gambar ke PIL Image
    img = image.squeeze().cpu().numpy() * 255
    img = img.astype(np.uint8)
    img_pil = Image.fromarray(img).convert('RGB')
    
    # Dapatkan ukuran gambar asli
    img_width, img_height = img_pil.size
    
    # Ubah ground truth dan predicted menjadi huruf kapital
    gt = gt.upper()
    pred = pred.upper()
    
    # Buat teks anotasi
    text = f"Ground Truth: {gt} | Predicted: {pred}"
    
    # Bagian font dikomentari karena tidak digunakan
    """
    # Tentukan font (coba Times New Roman, lalu fallback ke font lain)
    font_size = 15
    font = None
    font_paths = [
        "times.ttf",  # Times New Roman
        "arial.ttf",  # Arial sebagai fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Font default di banyak sistem Linux
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"Using font: {font_path}")
            break
        except:
            continue
    
    if font is None:
        print("Warning: Could not find any specified fonts. Using default font.")
        font = ImageFont.load_default()  # Gunakan font default jika semua font gagal
    
    # Hitung ukuran teks untuk memastikan tidak terpotong
    draw_temp = ImageDraw.Draw(img_pil)
    text_bbox = draw_temp.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Tentukan lebar minimum untuk gambar baru agar teks muat
    min_width = max(img_width, text_width + 40)  # Tambahkan padding 20 piksel di setiap sisi teks
    
    # Jika teks terlalu lebar, kurangi ukuran font hingga muat
    while text_width > min_width - 40 and font_size > 8:
        font_size -= 1
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
        text_bbox = draw_temp.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    
    # Tambahkan padding untuk area teks
    padding = 10
    text_area_height = text_height + 2 * padding
    
    # Buat gambar baru dengan lebar yang cukup dan ruang tambahan di atas untuk teks
    new_width = min_width
    new_height = img_height + text_area_height
    new_img = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))  # Latar belakang putih
    
    # Tempelkan gambar asli di bagian bawah, di tengah
    img_x = (new_width - img_width) // 2  # Posisi gambar di tengah
    new_img.paste(img_pil, (img_x, text_area_height))
    
    # Tambahkan teks "Ground Truth: ... | Predicted: ..."
    draw = ImageDraw.Draw(new_img)
    text_x = (new_width - text_width) // 2  # Posisi teks di tengah
    text_y = padding  # Posisi teks di area atas
    
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))  # Teks hitam
    """
    
    # Bagian penyimpanan gambar dikomentari
    """
    # Simpan gambar
    img_path = os.path.join(output_dir, f"{idx}_{gt}_to_{pred}.png")
    new_img.save(img_path)
    """

def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return None

def validation(model, criterion, evaluation_loader, converter, opt):
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    
    # Variabel untuk menghitung total kesalahan
    total_substitution = 0
    total_insertion = 0
    total_deletion = 0

    # Variabel untuk menghitung akurasi per karakter
    total_chars = 0  # Total karakter di semua ground truth
    correct_chars = 0  # Jumlah karakter yang benar
    char_counts = {}  # Menyimpan jumlah kemunculan setiap karakter
    char_correct = {}  # Menyimpan jumlah prediksi benar untuk setiap karakter

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data += batch_size
        image = image_tensors.to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time
            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for j, (gt, pred, pred_max_prob) in enumerate(zip(labels, preds_str, preds_max_prob)):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]
                pred_max_prob = pred_max_prob[:pred.find('[s]')]

            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            # Hitung akurasi per karakter
            gt = gt.upper()  # Ubah ke huruf kapital untuk konsistensi
            pred = pred.upper()

            # Inisialisasi jumlah kemunculan dan prediksi benar untuk setiap karakter di gt
            for char in gt:
                if char not in char_counts:
                    char_counts[char] = 0
                    char_correct[char] = 0
                char_counts[char] += 1

            # Bandingkan karakter per karakter
            min_len = min(len(gt), len(pred))
            total_chars += len(gt)
            for k in range(min_len):
                if gt[k] == pred[k]:
                    correct_chars += 1
                    char_correct[gt[k]] += 1
            # Jika pred lebih pendek dari gt, karakter yang hilang dianggap salah
            if len(pred) < len(gt):
                pass  # Karakter yang hilang sudah dihitung dalam total_chars
            # Jika pred lebih panjang dari gt, karakter tambahan diabaikan

            if pred == gt:
                n_correct += 1
            else:
                # Hitung jenis kesalahan
                sub, ins, dele = compute_error_types(gt, pred)
                total_substitution += sub
                total_insertion += ins
                total_deletion += dele

                # Bagian penyimpanan gambar dikomentari
                """
                # Simpan gambar berdasarkan jenis kesalahan
                if sub > 0:
                    save_image_with_annotation(image_tensors[j], gt, pred, "substitution", opt, i * batch_size + j)
                if ins > 0:
                    save_image_with_annotation(image_tensors[j], gt, pred, "insertion", opt, i * batch_size + j)
                if dele > 0:
                    save_image_with_annotation(image_tensors[j], gt, pred, "deletion", opt, i * batch_size + j)
                """

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)

    # Hitung akurasi per karakter secara keseluruhan
    if total_chars > 0:
        char_accuracy = (correct_chars / total_chars) * 100
    else:
        char_accuracy = 0.0

    # Hitung akurasi per karakter untuk setiap karakter
    char_accuracy_dict = {}
    for char in char_counts:
        if char_counts[char] > 0:
            char_accuracy_dict[char] = (char_correct[char] / char_counts[char]) * 100
        else:
            char_accuracy_dict[char] = 0.0

    # Buat char_stats sebagai dictionary yang berisi informasi akurasi per karakter
    char_stats = {
        'char_accuracy': char_accuracy,  # Akurasi per karakter keseluruhan
        'char_accuracy_dict': char_accuracy_dict,  # Akurasi per karakter
        'char_counts': char_counts,  # Jumlah kemunculan setiap karakter
        'char_correct': char_correct  # Jumlah prediksi benar untuk setiap karakter
    }

    # Bagian cetak akurasi per karakter dikomentari
    """
    # Cetak akurasi per karakter
    print(f"\nCharacter-Level Accuracy: {char_accuracy:.2f}%")
    print("Accuracy per Character:")
    for char, acc in sorted(char_accuracy_dict.items()):
        print(f"Character '{char}': {acc:.2f}% (Correct: {char_correct[char]}/{char_counts[char]})")
    """

    # Cetak jumlah kesalahan
    print(f"\nTotal Substitution Errors: {total_substitution}")
    print(f"Total Insertion Errors: {total_insertion}")
    print(f"Total Deletion Errors: {total_deletion}")

    # Tulis ke log file, pastikan direktori ada
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    with open(f'./result/{opt.exp_name}/error_counts.txt', 'a') as f:
        f.write(f"Total Substitution Errors: {total_substitution}\n")
        f.write(f"Total Insertion Errors: {total_insertion}\n")
        f.write(f"Total Deletion Errors: {total_deletion}\n")
        f.write(f"\nCharacter-Level Accuracy: {char_accuracy:.2f}%\n")
        f.write("Accuracy per Character:\n")
        for char, acc in sorted(char_accuracy_dict.items()):
            f.write(f"Character '{char}': {acc:.2f}% (Correct: {char_correct[char]}/{char_counts[char]})\n")

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data, char_stats

def test(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    print('loading pretrained model from %s' % opt.saved_model)
    # Tambahkan weights_only=True untuk keamanan
    model.load_state_dict(torch.load(opt.saved_model, map_location=device, weights_only=True))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])

    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    model.eval()
    with torch.no_grad():
        if opt.benchmark_all_eval:
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)
            log.write(eval_data_log)
            print(f'{accuracy_by_best_model:0.3f}')
            log.write(f'{accuracy_by_best_model:0.3f}\n')
            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
