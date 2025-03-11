import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

class ImageRenamer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Renamer")

        # Direktori gambar
        self.folder_path = "test"
        self.image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.current_index = 0
        self.renamed_count = 0  # Jumlah gambar yang sudah di-rename

        # Mendapatkan ukuran layar
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.preview_size = (int(screen_width * 0.4), int(screen_height * 0.4))  # 40% dari layar

        # Membuat widget
        self.create_widgets()
        
        # Load gambar pertama
        self.load_image()

        # Binding keyboard
        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('<Return>', self.save_rename)
        self.root.bind('<Escape>', self.quit_program)
        self.root.bind('<Delete>', self.delete_image)  # Tombol Delete pada keyboard

        # Ukuran window 60% dari layar & pusatkan di tengah layar
        window_width = int(screen_width * 0.6)
        window_height = int(screen_height * 0.6)
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

        self.root.attributes('-topmost', True)  # Selalu di atas

    def create_widgets(self):
        # Frame utama
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(expand=True)

        # Label untuk gambar
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.pack(pady=20)

        # Frame untuk input nama file
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(pady=15)

        # Label nama file
        self.filename_label = ttk.Label(self.input_frame, text="Nama File:", font=("Arial", 14, "bold"))
        self.filename_label.pack(side=tk.LEFT, padx=5)

        # Entry untuk rename
        self.filename_entry = ttk.Entry(self.input_frame, width=70, font=("Arial", 14), justify="center")
        self.filename_entry.pack(side=tk.LEFT, padx=5, ipady=5)

        # Frame untuk tombol navigasi
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        # Tombol navigasi
        self.prev_button = ttk.Button(self.button_frame, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=10)
        
        self.next_button = ttk.Button(self.button_frame, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=10)

        # **Tombol Delete**
        self.delete_button = ttk.Button(self.button_frame, text="Delete", command=self.delete_image, style="Red.TButton")
        self.delete_button.pack(side=tk.LEFT, padx=10)

        # Label info gambar (dengan progress)
        self.info_label = ttk.Label(self.main_frame, text="", font=("Arial", 12))
        self.info_label.pack(pady=10)

        # **Style untuk tombol Delete (Warna Merah)**
        style = ttk.Style()
        style.configure("Red.TButton", foreground="red", font=("Arial", 12, "bold"))

    def load_image(self):
        if not self.image_files:
            self.image_label.config(text="No images found", image="")
            self.filename_entry.delete(0, tk.END)
            self.info_label.config(text="")
            return

        # Load gambar
        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.folder_path, current_file)
        image = Image.open(image_path)
        
        # Resize gambar agar maksimal 40% dari layar dengan menjaga aspek rasio
        orig_width, orig_height = image.size
        max_width, max_height = self.preview_size
        scale = min(max_width / orig_width, max_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Menyimpan referensi
        
        # Update entry dengan nama file tanpa ekstensi
        self.filename_entry.delete(0, tk.END)
        name_without_ext, ext = os.path.splitext(current_file)  # Pisahkan nama & ekstensi
        if '_' in name_without_ext:
            prefix, suffix = name_without_ext.rsplit('_', 1)
            self.prefix = prefix + '_'
            self.filename_entry.insert(0, suffix)
        else:
            self.prefix = ''
            self.filename_entry.insert(0, name_without_ext)

        self.current_ext = ext  # Simpan ekstensi agar tidak berubah saat rename

        # **Update info label dengan progress**
        self.update_progress_label()

    def save_rename(self, event=None):
        if not self.image_files:
            return

        old_name = self.image_files[self.current_index]
        new_suffix = self.filename_entry.get().strip()  # Hapus spasi berlebih
        new_name = self.prefix + new_suffix if self.prefix else new_suffix
        
        # Tambahkan kembali ekstensi yang asli
        new_name += self.current_ext

        # Rename file
        old_path = os.path.join(self.folder_path, old_name)
        new_path = os.path.join(self.folder_path, new_name)
        
        if old_name != new_name and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            self.image_files[self.current_index] = new_name
            self.info_label.configure(text=f"Renamed to: {new_name}")

            # **Tambah counter rename**
            self.renamed_count += 1
        else:
            self.info_label.configure(text="No changes made")

        # **Update progress label setelah rename**
        self.update_progress_label()

    def delete_image(self, event=None):
        """Menghapus gambar saat ini setelah konfirmasi."""
        if not self.image_files:
            return
        
        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.folder_path, current_file)

        # Konfirmasi sebelum menghapus
        confirm = messagebox.askyesno("Delete Image", f"Are you sure you want to delete {current_file}?")
        if confirm:
            os.remove(image_path)
            self.image_files.pop(self.current_index)

            # Jika masih ada gambar lain, load gambar berikutnya
            if self.image_files:
                self.current_index = min(self.current_index, len(self.image_files) - 1)
                self.load_image()
            else:
                # Jika tidak ada gambar tersisa
                self.image_label.config(text="No images left", image="")
                self.filename_entry.delete(0, tk.END)
                self.info_label.config(text="")

    def prev_image(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self, event=None):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def quit_program(self, event=None):
        self.root.quit()
        self.root.destroy()

    def update_progress_label(self):
        """Update progress label dengan format: 'Image X/Y | Renamed: A'."""
        progress_text = f"Image {self.current_index + 1}/{len(self.image_files)} | Renamed: {self.renamed_count}"
        self.info_label.configure(text=progress_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRenamer(root)
    root.mainloop()
