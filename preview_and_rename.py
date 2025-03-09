import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImagePreviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Previewer")
        self.root.geometry("1400x1000")

        self.image_list = []
        self.current_index = 0
        self.folder_path = ""

        self.canvas = tk.Canvas(root, width=1200, height=800)
        self.canvas.pack(pady=10)

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        self.filename_label = tk.Label(self.control_frame, text="", font=("Arial", 14))
        self.filename_label.pack(side=tk.LEFT, padx=5)

        self.rename_entry = tk.Entry(self.control_frame, font=("Arial", 14), width=40)
        self.rename_entry.pack(side=tk.LEFT, padx=5)
        self.rename_entry.bind("<Return>", self.rename_image_with_event)

        self.rename_btn = tk.Button(self.control_frame, text="Rename", 
                                  command=self.rename_image, font=("Arial", 12))
        self.rename_btn.pack(side=tk.LEFT, padx=5)

        # Tombol Delete baru
        self.delete_btn = tk.Button(self.control_frame, text="Delete", 
                                  command=self.delete_image, font=("Arial", 12))
        self.delete_btn.pack(side=tk.LEFT, padx=5)

        self.prev_btn = tk.Button(self.control_frame, text="Previous", 
                                  command=self.previous_image, font=("Arial", 12))
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(self.control_frame, text="Next", 
                                  command=self.next_image, font=("Arial", 12))
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.select_btn = tk.Button(root, text="Pilih Folder Dataset", 
                                  command=self.select_folder, font=("Arial", 12))
        self.select_btn.pack(pady=10)

        self.progress_label = tk.Label(root, text="", font=("Arial", 12))
        self.progress_label.pack(pady=5)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory(title="Pilih Folder Dataset")
        if self.folder_path:
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_list = [f for f in os.listdir(self.folder_path) 
                             if f.lower().endswith(valid_extensions)]
            self.image_list.sort()
            
            if self.image_list:
                self.current_index = 0
                self.show_image()
            else:
                messagebox.showinfo("Info", "Tidak ada gambar ditemukan di folder ini")

    def show_image(self):
        if not self.image_list:
            self.canvas.delete("all")  # Bersihkan canvas jika tidak ada gambar
            self.filename_label.config(text="")
            self.rename_entry.delete(0, tk.END)
            self.progress_label.config(text="")
            return

        image_path = os.path.join(self.folder_path, self.image_list[self.current_index])
        
        img = Image.open(image_path)
        img = self.resize_image(img, (1200, 800))
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(600, 400, image=self.photo)
        
        self.filename_label.config(text=self.image_list[self.current_index])
        self.rename_entry.delete(0, tk.END)
        self.rename_entry.insert(0, os.path.splitext(self.image_list[self.current_index])[0])
        self.progress_label.config(text=f"{self.current_index + 1}/{len(self.image_list)}")

    def resize_image(self, img, max_size):
        width, height = img.size
        max_width, max_height = max_size
        
        ratio = min(max_width/width, max_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def previous_image(self):
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image()

    def rename_image(self):
        if not self.image_list:
            return

        old_name = self.image_list[self.current_index]
        new_name = self.rename_entry.get()
        
        old_ext = os.path.splitext(old_name)[1]
        if not new_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            new_name += old_ext

        try:
            old_path = os.path.join(self.folder_path, old_name)
            new_path = os.path.join(self.folder_path, new_name)
            
            os.rename(old_path, new_path)
            self.image_list[self.current_index] = new_name
            self.filename_label.config(text=new_name)
            self.next_image()  # Pindah ke gambar berikutnya setelah rename
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal merename file: {str(e)}")

    def rename_image_with_event(self, event):
        self.rename_image()

    def delete_image(self):
        if not self.image_list:
            return

        # Konfirmasi sebelum menghapus
        if not messagebox.askyesno("Konfirmasi", "Apakah Anda yakin ingin menghapus gambar ini?"):
            return

        current_image = self.image_list[self.current_index]
        image_path = os.path.join(self.folder_path, current_image)

        try:
            # Hapus file dari folder
            os.remove(image_path)
            # Hapus dari daftar
            del self.image_list[self.current_index]
            
            # Sesuaikan index
            if self.image_list:
                if self.current_index >= len(self.image_list):
                    self.current_index = len(self.image_list) - 1
                self.show_image()
            else:
                self.show_image()  # Akan membersihkan canvas jika tidak ada gambar tersisa
                
            messagebox.showinfo("Sukses", f"Gambar {current_image} berhasil dihapus")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menghapus gambar: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePreviewer(root)
    root.mainloop()