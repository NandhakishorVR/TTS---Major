import torch
import matplotlib.pyplot as plt
import os

# Path to the spectrograms folder
spectrogram_dir = r"C:\Users\DELL\Downloads\ttsnew\LJSpeech-1.1\spectrograms"

# List all spectrogram files
spectrogram_files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.pt')]

if not spectrogram_files:
    print("❌ No spectrograms found! Check if `preprocess_ljspeech.py` ran correctly.")
else:
    # Load a random spectrogram file
    mel_file = os.path.join(spectrogram_dir, spectrogram_files[0])  # Load the first file
    mel_spec = torch.load(mel_file)  # Load tensor (shape: [1, 80, TimeSteps])

    # Remove batch dimension if needed (convert shape [1, 80, TimeSteps] -> [80, TimeSteps])
    if len(mel_spec.shape) == 3:
        mel_spec = mel_spec.squeeze(0)  # Removes the first dimension

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bin")
    plt.title(f"Mel Spectrogram - {spectrogram_files[0]}")
    plt.show()
