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
import editdistance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_error_types(gt, pred):
    """Calculate substitution, insertion, and deletion errors between ground truth and prediction."""
    edit_ops = editdistance.eval(gt, pred, return_ops=True)[1]
    
    substitutions = 0
    insertions = 0
    deletions = 0
    
    for op, _, _ in edit_ops:
        if op == 'replace':
            substitutions += 1
        elif op == 'insert':
            insertions += 1
        elif op == 'delete':
            deletions += 1
            
    return substitutions, insertions, deletions

def save_error_image(image, gt, pred, error_type, save_dir, idx):
    """Save image with ground truth and predicted text annotated."""
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().squeeze()
        if image.ndim == 3:  # If RGB
            image = image.transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        img = Image.fromarray(image)
    else:
        img = image
    
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
    
    draw.text((5, 5), f"GT: {gt}", fill=(255, 0, 0), font=font)
    draw.text((5, 25), f"Pred: {pred}", fill=(0, 255, 0), font=font)
    
    img_path = os.path.join(save_dir, f"{error_type}_{idx}.png")
    img_copy.save(img_path)

def validation(model, criterion, evaluation_loader, converter, opt):
    """Validation or evaluation with error type analysis."""
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    
    base_save_dir = f'./result/{opt.exp_name}/error_images'
    sub_dir = os.path.join(base_save_dir, 'substitutions')
    ins_dir = os.path.join(base_save_dir, 'insertions')
    del_dir = os.path.join(base_save_dir, 'deletions')

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
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
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
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

        for idx, (gt, pred, img_tensor) in enumerate(zip(labels, preds_str, image_tensors)):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]

            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1
            else:
                subs, ins, dels = calculate_error_types(gt, pred)
                total_substitutions += subs
                total_insertions += ins
                total_deletions += dels

                if subs > 0:
                    save_error_image(img_tensor, gt, pred, 'substitution', sub_dir, f"{i}_{idx}")
                if ins > 0:
                    save_error_image(img_tensor, gt, pred, 'insertion', ins_dir, f"{i}_{idx}")
                if dels > 0:
                    save_error_image(img_tensor, gt, pred, 'deletion', del_dir, f"{i}_{idx}")

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)

    print(f"Total Substitution Errors: {total_substitutions}")
    print(f"Total Insertion Errors: {total_insertions}")
    print(f"Total Deletion Errors: {total_deletions}")

    with open(f'./result/{opt.exp_name}/error_stats.txt', 'a') as f:
        f.write(f"Total Substitution Errors: {total_substitutions}\n")
        f.write(f"Total Insertion Errors: {total_insertions}\n")
        f.write(f"Total Deletion Errors: {total_deletions}\n")

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, [], labels, infer_time, length_of_data

def test(opt):
    """ model configuration """
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
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
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
