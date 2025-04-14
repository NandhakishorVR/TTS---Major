import os
import torch
import matplotlib.pyplot as plt

# Path to the generated spectrograms folder
output_dir = os.path.join("ttsnew", "generated_spectrograms")

def visualize_spectrogram(file_name="hello_gokul.pt"):
    file_path = os.path.join(output_dir, file_name)

    if not os.path.exists(file_path):
        print(f"❌ Spectrogram file not found at {file_path}")
        return

    # Load the spectrogram
    spectrogram = torch.load(file_path)

    # Remove batch dimension if needed (convert shape [1, 80, TimeSteps] -> [80, TimeSteps])
    if len(spectrogram.shape) == 3:
        spectrogram = spectrogram.squeeze(0)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bin")
    plt.title(f"Mel Spectrogram - {file_name}")
    plt.show()

# Example visualization
visualize_spectrogram("hello_gokul.pt")
