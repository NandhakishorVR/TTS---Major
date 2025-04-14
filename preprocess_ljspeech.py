import os
import torch
import librosa
import numpy as np
import torchaudio.transforms as transforms
import pandas as pd

# Define paths
DATASET_PATH = r"C:\Users\DELL\Downloads\ttsnew\LJSpeech-1.1"
wav_dir = os.path.join(DATASET_PATH, "wavs")
mel_spectrogram_dir = os.path.join(DATASET_PATH, "melspectrograms")

# Create directory if it doesn't exist
os.makedirs(mel_spectrogram_dir, exist_ok=True)

# Load metadata
metadata_path = os.path.join(DATASET_PATH, "metadata_phonemes.csv")
metadata = pd.read_csv(metadata_path)

# Function to convert wav to mel spectrogram
def wav_to_mel_spectrogram(wav_path):
    waveform, sr = librosa.load(wav_path, sr=22050)
    mel_transform = transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=1024, hop_length=256)
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)
    mel_spectrogram = mel_transform(waveform_tensor)
    return mel_spectrogram.squeeze(0)

# Process and save mel spectrograms
for index, row in metadata.iterrows():
    filename = row['filename']
    wav_path = os.path.join(wav_dir, filename + '.wav')
    spectrogram_path = os.path.join(mel_spectrogram_dir, filename + '.pt')

    if not os.path.exists(wav_path):
        print(f" Missing WAV file: {filename}.wav")
        continue

    mel_spectrogram = wav_to_mel_spectrogram(wav_path)
    
    # Save spectrogram as PyTorch tensor
    torch.save(mel_spectrogram, spectrogram_path)
    print(f" Saved: {spectrogram_path}")

print(" Mel spectrogram generation complete!")
