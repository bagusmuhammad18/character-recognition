import os
import glob
from collections import Counter
import csv

def count_characters_after_underscore(folder_path):
    """
    Menghitung karakter A-Z dan 0-9 setelah underscore pada nama file di folder tertentu
    """
    # Ambil semua file gambar (jpg, png, jpeg)
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    image_files += glob.glob(os.path.join(folder_path, "*.png"))
    image_files += glob.glob(os.path.join(folder_path, "*.jpeg"))
    
    if not image_files:
        print(f"Tidak ada gambar di folder {folder_path}")
        return Counter()
    
    # Counter untuk karakter
    char_counter = Counter()
    
    # Proses setiap file
    for file_path in image_files:
        # Ambil nama file tanpa path dan ekstensi
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Split berdasarkan underscore
        parts = file_name.split('_')
        if len(parts) < 2:
            print(f"File {file_name} tidak memiliki underscore, dilewati")
            continue
        
        # Ambil bagian setelah underscore
        relevant_part = parts[1]
        
        # Hitung hanya karakter A-Z dan 0-9
        for char in relevant_part:
            if char.isalnum():  # Hanya A-Z, a-z, 0-9
                if char.isalpha():
                    char = char.upper()  # Ubah ke huruf kapital
                char_counter[char] += 1
    
    return char_counter

def save_to_csv(counter, folder_name, csv_writer):
    """
    Menyimpan hasil perhitungan ke file CSV
    """
    if counter:
        for char, count in sorted(counter.items()):
            csv_writer.writerow([folder_name, char, count])
    else:
        csv_writer.writerow([folder_name, "No data", 0])

def main():
    # Definisikan folder untuk train, test, dan val
    folders = {
        "train": "train",
        "test": "test",
        "val": "val"
    }
    
    # Buat folder jika belum ada
    for split in folders:
        path = folders[split]
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Membuat folder: {path}")
    
    # Nama file CSV untuk menyimpan hasil
    output_file = "character_count_results.csv"
    
    # Counter total untuk semua karakter
    total_counter = Counter()
    
    # Buka file CSV untuk menulis
    with open(output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        
        # Tulis header
        csv_writer.writerow(["Folder", "Character", "Count"])
        
        # Proses setiap folder
        for split in folders:
            print(f"\nMemproses folder {split}...")
            folder_path = folders[split]
            
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} tidak ditemukan")
                csv_writer.writerow([split, "Folder not found", 0])
                continue
            
            # Hitung karakter di folder ini
            char_counter = count_characters_after_underscore(folder_path)
            
            # Tulis hasil ke CSV
            save_to_csv(char_counter, split, csv_writer)
            
            # Tambahkan ke counter total
            total_counter.update(char_counter)
        
        # Tulis hasil total
        if total_counter:
            csv_writer.writerow([])  # Baris kosong untuk pemisah
            csv_writer.writerow(["Total", "", ""])  # Header untuk total
            save_to_csv(total_counter, "Total", csv_writer)
        else:
            csv_writer.writerow(["Total", "No data", 0])
    
    print(f"\nSemua hasil telah disimpan ke {output_file}")

if __name__ == "__main__":
    main()