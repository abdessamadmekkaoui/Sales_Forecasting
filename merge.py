import pandas as pd
import chardet
import os

def detect_encoding(file_path, num_bytes=10000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    return result['encoding']

# Folder containing your CSV files
folder_path = 'C:/Users/mekka/Downloads/NEW DATA/'  # change this to your actual folder
output_file = 'merged.csv'
chunksize = 10**6

# Get all CSV file paths
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
    for i, file_path in enumerate(csv_files):
        encoding = detect_encoding(file_path)
        print(f"Merging {file_path} with encoding: {encoding}")
        reader = pd.read_csv(file_path, encoding=encoding, chunksize=chunksize)
        for j, chunk in enumerate(reader):
            write_header = (i == 0 and j == 0)
            chunk.to_csv(f_out, index=False, header=write_header, mode='a')

