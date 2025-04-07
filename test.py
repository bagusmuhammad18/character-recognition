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
    # Using editdistance to get the detailed operations
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
    
    # Convert tensor to PIL image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().squeeze()
        if image.ndim == 3:  # If RGB
            image = image.transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        img = Image.fromarray(image)
    else:
        img = image
    
    # Create a copy to draw on
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a default font, fall back to basic if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((5, 5), f"GT: {gt}", fill=(255, 0, 0), font=font)
    draw.text((5, 25), f"Pred: {pred}", fill=(0, 255, 0), font=font)
    
    # Save image
    img_path = os.path.join(save_dir, f"{error_type}_{idx}.png")
    img_copy.save(img_path)

def validation(model, criterion, evaluation_loader, converter, opt):
    """Validation or evaluation with error type analysis."""
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    
    # Counters for error types
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    
    # Base directory for saving error images
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
                # Calculate error types
                subs, ins, dels = calculate_error_types(gt, pred)
                total_substitutions += subs
                total_insertions += ins
                total_deletions += dels

                # Save images for each error type if present
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

    # Print error statistics
    print(f"Total Substitution Errors: {total_substitutions}")
    print(f"Total Insertion Errors: {total_insertions}")
    print(f"Total Deletion Errors: {total_deletions}")

    with open(f'./result/{opt.exp_name}/error_stats.txt', 'a') as f:
        f.write(f"Total Substitution Errors: {total_substitutions}\n")
        f.write(f"Total Insertion Errors: {total_insertions}\n")
        f.write(f"Total Deletion Errors: {total_deletions}\n")

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, [], labels, infer_time, length_of_data

# Rest of the code (benchmark_all_eval, test, and main) remains largely unchanged
# Just ensure to update the validation call in benchmark_all_eval and test functions if needed

def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    # ... (previous code remains the same until validation call)
    _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
        model, criterion, evaluation_loader, converter, opt)
    # ... (rest of the function remains the same)

def test(opt):
    # ... (previous code remains the same until validation call)
    if not opt.benchmark_all_eval:
        # ... (previous code)
        _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
            model, criterion, evaluation_loader, converter, opt)
        # ... (rest of the function)

if __name__ == '__main__':
    # ... (parser and main code remains unchanged)
    test(opt)
