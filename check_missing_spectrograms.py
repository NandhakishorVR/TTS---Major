import os

spectrogram_dir = r"C:\Users\DELL\Downloads\ttsnew\LJSpeech-1.1\melspectrograms"
metadata_path = r"C:\Users\DELL\Downloads\ttsnew\LJSpeech-1.1\metadata_phonemes.csv"

# Load metadata to check expected filenames
import pandas as pd
metadata = pd.read_csv(metadata_path)

missing_files = []
for filename in metadata['filename']:
    spectrogram_path = os.path.join(spectrogram_dir, filename + '.pt')
    if not os.path.exists(spectrogram_path):
        missing_files.append(filename)

if missing_files:
    print(f" Missing {len(missing_files)} spectrogram files. Example: {missing_files[:5]}")
else:
    print(" All spectrograms are present!")
