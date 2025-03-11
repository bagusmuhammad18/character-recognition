import os

# Dataset yang akan diproses
datasets = ['train', 'val', 'test']

# Fungsi untuk mendapatkan plat nomor dari nama file
def get_plate_number(filename):
    if '_' in filename:
        return filename.split('_')[1].split('.')[0]  # Ambil setelah underscore, sebelum ekstensi
    else:
        return filename.split('.')[0]  # Ambil nama sebelum ekstensi jika tidak ada underscore

# Proses setiap dataset
for dataset in datasets:
    folder_path = dataset  # Langsung pakai nama folder tanpa titik
    output_file = f'{dataset}.txt'

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' tidak ditemukan, dilewati...")
        continue

    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.jpg'):
                file_path = f"{dataset}/{filename}"  # Format path tanpa titik
                plate_number = get_plate_number(filename)
                f.write(f"{file_path}\t{plate_number}\n")

    print(f"File '{output_file}' telah dibuat.")

