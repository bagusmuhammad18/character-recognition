...
# Tambahkan setelah line "confidence_score_list.append(confidence_score)"
# dan pastikan kamu import PIL di atas: from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont

substitution_dir = f'./result/{opt.exp_name}/substitution'
insertion_dir = f'./result/{opt.exp_name}/insertion'
delete_dir = f'./result/{opt.exp_name}/deletion'
os.makedirs(substitution_dir, exist_ok=True)
os.makedirs(insertion_dir, exist_ok=True)
os.makedirs(delete_dir, exist_ok=True)

for idx_in_batch in range(image.size(0)):
    pred_text = preds_str[idx_in_batch]
    gt_text = labels[idx_in_batch]

    if pred_text == gt_text:
        continue

    # convert tensor to PIL image
    img_np = image[idx_in_batch].detach().cpu().numpy()
    img_np = (img_np * 0.5 + 0.5) * 255  # de-normalize
    img_np = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)
    img_pil = Image.fromarray(img_np).convert("RGB")

    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    draw.text((2, 2), f"GT: {gt_text}", fill=(0, 255, 0), font=font)
    draw.text((2, 18), f"Pred: {pred_text}", fill=(255, 0, 0), font=font)

    # Compute edit operations
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, gt_text, pred_text)
    ops = matcher.get_opcodes()

    # classify errors
    has_sub = any(op[0] == 'replace' for op in ops)
    has_ins = any(op[0] == 'insert' for op in ops)
    has_del = any(op[0] == 'delete' for op in ops)

    base_name = f'{i:05}_{idx_in_batch:02}_{gt_text}_{pred_text}.png'
    if has_sub:
        img_pil.save(os.path.join(substitution_dir, base_name))
    if has_ins:
        img_pil.save(os.path.join(insertion_dir, base_name))
    if has_del:
        img_pil.save(os.path.join(delete_dir, base_name))
