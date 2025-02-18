import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, TextBox

# Folder dataset
DATASET_FOLDER = 'dataset_v1/train'

# Daftar file gambar
image_files = [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
if not image_files:
    raise FileNotFoundError("Tidak ditemukan gambar di folder dataset_v1!")

# Parameter awal
current_index = 0
kernel_size = 5
offset = 0.2

def load_and_preprocess_image(filename, kernel_size, offset):
    """Memuat dan memproses gambar dengan preprocessing grayscale dan binarisasi."""
    img = cv2.imread(os.path.join(DATASET_FOLDER, filename))
    if img is None:
        raise FileNotFoundError(f"Gambar {filename} tidak dapat dibuka.")
    
    # Konversi ke grayscale jika 3 channel
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Adaptive thresholding berbasis mean lokal
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Pastikan ganjil
    local_mean = cv2.blur(img_gray, (kernel_size, kernel_size))
    binary_img = (img_gray > (local_mean - int(offset * 255))).astype(np.uint8) * 255

    return img_gray, binary_img

def update_display():
    """Memperbarui tampilan gambar."""
    k = int(slider_kernel.val)
    o = slider_offset.val
    img_gray, img_bin = load_and_preprocess_image(image_files[current_index], k, o)
    ax1.imshow(img_gray, cmap='gray')
    ax1.set_title(f'Grayscale: {image_files[current_index]}')
    ax2.imshow(img_bin, cmap='gray')
    ax2.set_title(f'Binarized (Kernel={k}, Offset={o:.2f})')
    fig.canvas.draw_idle()

def on_key(event):
    """Navigasi gambar dengan panah kiri dan kanan."""
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(image_files)
        update_display()
    elif event.key == 'left':
        current_index = (current_index - 1) % len(image_files)
        update_display()

def update_kernel_text(text):
    """Memperbarui kernel size dari input teks."""
    try:
        val = int(text)
        if val % 2 == 1 and 3 <= val <= 21:
            slider_kernel.set_val(val)
            update_display()
        else:
            print("Kernel size harus ganjil dan antara 3 hingga 21.")
    except ValueError:
        print("Input kernel size tidak valid.")

def update_offset_text(text):
    """Memperbarui offset dari input teks."""
    try:
        val = float(text)
        if 0.0 <= val <= 1.0:
            slider_offset.set_val(val)
            update_display()
        else:
            print("Offset harus antara 0.0 dan 1.0.")
    except ValueError:
        print("Input offset tidak valid.")

# Setup tampilan dengan Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.35)

# Menampilkan gambar pertama
img_gray, img_bin = load_and_preprocess_image(image_files[current_index], kernel_size, offset)
ax1.imshow(img_gray, cmap='gray')
ax1.set_title(f'Grayscale: {image_files[current_index]}')
ax2.imshow(img_bin, cmap='gray')
ax2.set_title(f'Binarized (Kernel={kernel_size}, Offset={offset})')

# Tambahkan slider untuk kernel size dan offset
ax_slider_kernel = plt.axes([0.2, 0.25, 0.6, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_offset = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor='lightgoldenrodyellow')

slider_kernel = Slider(ax_slider_kernel, 'Kernel Size', 3, 21, valinit=kernel_size, valstep=2)
slider_offset = Slider(ax_slider_offset, 'Offset', 0.0, 1.0, valinit=offset)

# Tambahkan input teks untuk kernel size dan offset di bawah slider
ax_text_kernel = plt.axes([0.2, 0.13, 0.2, 0.04])
ax_text_offset = plt.axes([0.6, 0.13, 0.2, 0.04])

text_kernel = TextBox(ax_text_kernel, 'Input Kernel:', initial=str(kernel_size))
text_offset = TextBox(ax_text_offset, 'Input Offset:', initial=str(offset))

# Hubungkan event slider dan input teks
slider_kernel.on_changed(lambda val: update_display())
slider_offset.on_changed(lambda val: update_display())
text_kernel.on_submit(update_kernel_text)
text_offset.on_submit(update_offset_text)

# Hubungkan event keyboard untuk navigasi gambar
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
