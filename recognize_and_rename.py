import string
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import os
import shutil
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(opt):
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

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # Get list of images
    image_list = [os.path.join(opt.image_folder, img) for img in os.listdir(opt.image_folder) 
                 if img.endswith(('.jpg', '.png', '.jpeg'))]
    image_list.sort()
    current_idx = 0

    if not image_list:
        print("No images found in the specified folder.")
        return

    # Counter for images with confidence > 0.5
    high_confidence_count = 0
    total_images = len(image_list)

    # Create destination folder
    output_folder = "hitam di atas 0.5"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Predict and process
    model.eval()
    with torch.no_grad():
        while True:
            # Load current image
            img_path = image_list[current_idx]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                current_idx += 1
                if current_idx >= len(image_list):
                    break
                continue

            # Get prediction
            pred = ""
            confidence_score = 0.0
            
            for image_tensors, image_path_list in demo_loader:
                if image_path_list[0] != img_path:
                    continue
                    
                image = image_tensors.to(device)
                length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
                text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)])
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, preds_size)
                else:
                    preds = model(image, text_for_pred, is_train=False)
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                
                pred = preds_str[0]
                pred_max_prob = preds_max_prob[0]
                
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                pred = pred.upper()
                confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                
                # Count and copy images with confidence > 0.5
                if confidence_score > 0.5:
                    high_confidence_count += 1
                    # Create new filename using prediction
                    original_ext = os.path.splitext(img_path)[1]
                    new_filename = f"{pred}{original_ext}"
                    dest_path = os.path.join(output_folder, new_filename)
                    
                    # Handle potential duplicate filenames
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{pred}_{counter}{original_ext}"
                        dest_path = os.path.join(output_folder, new_filename)
                        counter += 1
                    
                    shutil.copy2(img_path, dest_path)
                
                break

            # Clear terminal and show results
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f'Image: {os.path.basename(img_path)}')
            print(f'Prediction: {pred}')
            print(f'Confidence Score: {confidence_score:.4f}')
            print(f'Images with confidence > 0.5: {high_confidence_count}/{total_images}')
            if confidence_score > 0.5:
                print(f'Copied as: {os.path.basename(dest_path)}')
            print('\nProcessing next image...')

            # Automatic navigation
            if current_idx < len(image_list) - 1:
                current_idx += 1
            else:
                break

    # Final summary
    os.system('clear' if os.name == 'posix' else 'cls')
    print(f'\nProcessing complete!')
    print(f'Total images processed: {total_images}')
    print(f'Images with confidence score > 0.5: {high_confidence_count}')
    print(f'Percentage: {(high_confidence_count/total_images)*100:.2f}%')
    print(f'Images copied to: {output_folder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default="/home/bagus/Downloads/Dataset Kaggle/dataset2/Indonesian License Plate Recognition Dataset/images/hitam", help='path to folder containing images to be recognized')
    parser.add_argument('--saved_model', default="saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111 16 Feb 2025 16:07 (no preproc) (scientific_maltodon)/best_accuracy.pth", help='path to the pretrained model')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum label length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points for TPS transformation')
    parser.add_argument('--input_channel', type=int, default=1, help='input channel for images')
    parser.add_argument('--output_channel', type=int, default=512, help='output channel for feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the LSTM hidden state')
    opt = parser.parse_args()

    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt) 