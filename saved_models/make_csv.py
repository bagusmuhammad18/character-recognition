import re
import csv

# Path file log dan file CSV keluaran
folder = "TPS-ResNet-BiLSTM-Attn-Seed1111 16 Feb 2025 14:54 (preproc grayscale only) (armed_altitude)"
log_file = f"{folder}/log_train.txt"
csv_file = f"{folder}/log_train.csv"

data = []

# Baca file log
with open(log_file, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    # Cari baris yang mengandung informasi training, misal: "[1/3000] Train loss: ..."
    if line.startswith('['):
        # Ekstrak iteration, train loss, valid loss, dan elapsed time (jika diperlukan)
        match = re.search(
            r'\[(\d+)/(\d+)\]\s+Train loss:\s*([\d\.]+),\s*Valid loss:\s*([\d\.]+),\s*Elapsed_time:\s*([\d\.]+)',
            line
        )
        if match:
            iteration = int(match.group(1))
            train_loss = float(match.group(3))
            valid_loss = float(match.group(4))
            # Jika diperlukan: elapsed_time = float(match.group(5))

            # Baris berikutnya seharusnya mengandung "Current_accuracy" dan "Current_norm_ED"
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                match2 = re.search(
                    r'Current_accuracy\s*:\s*([\d\.]+),\s*Current_norm_ED\s*:\s*([\d\.]+)',
                    next_line
                )
                if match2:
                    accuracy = float(match2.group(1))
                    norm_ED = float(match2.group(2))
                    # Simpan data ke dalam list
                    data.append({
                        "iteration": iteration,
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "accuracy": accuracy,
                        "norm_ED": norm_ED
                    })
                    # Lewati baris berikutnya karena sudah diproses
                    i += 1
    i += 1

# Tulis data ke file CSV
with open(csv_file, 'w', newline='') as f:
    fieldnames = ["iteration", "train_loss", "valid_loss", "accuracy", "norm_ED"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"CSV file '{csv_file}' telah berhasil dibuat.")
