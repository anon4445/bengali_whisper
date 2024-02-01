import pandas as pd
import os
import shutil
from tqdm import tqdm

datasets = {
    'train': {'tsv': 'train.tsv', 'dest_folder': 'train'},
    'valid': {'tsv': 'valid.tsv', 'dest_folder': 'valid'},
    'test': {'tsv': 'test.tsv', 'dest_folder': 'test'}
}

audio_folder = 'clips'
for dataset in datasets.values():
    if not os.path.exists(dataset['dest_folder']):
        os.makedirs(dataset['dest_folder'])
    df = pd.read_csv(dataset['tsv'], sep='\t')
    for filename in tqdm(df['path'], desc=f"Copying {dataset['tsv']} files"):
        try:
            full_path = os.path.join(audio_folder, filename)
            if os.path.exists(full_path):
                shutil.copy(full_path, dataset['dest_folder'])
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
print("Operation completed.")
