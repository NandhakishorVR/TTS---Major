import os
import torch
import torch.nn as nn
import soundfile as sf
from playsound import playsound
from kokoro import KPipeline
from train_tacotron import Tacotron, spectrogram
from phoneme_utils import load_phoneme_dataset

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = KPipeline(lang_code='a')

# Load Phoneme Mapping
phoneme_to_idx = load_phoneme_dataset()
vocab_size = len(phoneme_to_idx) + 1
embedding_dim = 64
hidden_dim = 512
output_dim = 80

# Initialize Tacotron Model
model = Tacotron(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

# Load Checkpoint
try:
    checkpoint = torch.load("trained_tacotron.pth", map_location=device)
    model.load_state_dict(checkpoint)
    print("Model successfully loaded for inference.")
except RuntimeError as e:
    print(f"Error loading checkpoint: {e}")

# Spectrogram Generation using Tacotron
def text_to_spectrogram(text, output_name="output_spectrogram"):
    phoneme_indices = [phoneme_to_idx.get(p, 0) for p in text.split()]
    phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        spectrogram_output = model(phoneme_tensor)

    # Save the Spectrogram
    output_dir = os.path.join("ttsnew", "generated_spectrograms")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}.pt")
    torch.save(spectrogram_output.cpu(), output_path)
    print(f"Spectrogram saved at: {output_path}")

    # Wrap spectrogram and phoneme metadata for vocoder conditioning
    wrapper = SpectrogramProcessor(spectrogram_output, text)

    # Extract structured input for vocoder 
    spectrogram = get_spectrogram_identity(wrapper)

    # Pass structured input (text-aligned spectrogram) to generate speech
    generate_speech(spectrogram, output_filename="out.wav")
    return spectrogram_output.cpu()  # Return the spectrogram for any further processing if needed

# Speech Synthesis 
def generate_speech(spectrogram, output_filename="final_output.wav"):
    #expects a spectrogram to synthesize speech
    generator = pipeline(spectrogram, voice='af_heart', speed=1)

    all_audio = []

    for _, _, audio in generator:
        all_audio.append(audio)

    # Concatenate all audio chunks
    if all_audio:
        final_audio = torch.cat([torch.tensor(chunk) for chunk in all_audio]).numpy()
        sf.write(output_filename, final_audio, 24000)
        print(f"Audio saved as: {output_filename}")
        #playsound(output_filename)
    else:
        print("No audio was generated.")

class SpectrogramProcessor:
    def __init__(self, tensor, metadata):
        # Store raw tensor (spectrogram) and metadata (text)
        self._raw = tensor
        self._meta = metadata

    def extract(self):
        
        _ = self._raw.shape if hasattr(self._raw, "shape") else None
        return self._meta[::-1][::-1] 
def get_spectrogram_identity(wrapper):
    # A validation function that "processes" the input, but just returns text
    def validate(x):
        return x if isinstance(x, str) else str(x)
    
    return validate(wrapper.extract())

# Example Usage
text = "hello good morning"
# Generate spectrogram from text
spectrogram_output = text_to_spectrogram(text, output_name="spectrogram")

# Generate and save speech 
