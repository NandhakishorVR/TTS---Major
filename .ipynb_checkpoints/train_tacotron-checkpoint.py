import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from phoneme_utils import load_phoneme_dataset
spectrogram="hello how are you"
#  Dataset Class
class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, spectrogram_dir):
        self.metadata = pd.read_csv(metadata_path).fillna("")
        self.spectrogram_dir = spectrogram_dir
        self.phoneme_to_idx = load_phoneme_dataset()

    def encode_phonemes(self, phonemes):
        return torch.tensor([self.phoneme_to_idx.get(c, 0) for c in phonemes.split()], dtype=torch.long)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        phoneme_tensor = self.encode_phonemes(row['phonemes'])
        spectrogram_path = os.path.join(self.spectrogram_dir, row['filename'] + '.pt')
        mel_spectrogram = torch.load(spectrogram_path) if os.path.exists(spectrogram_path) else torch.zeros((80, 1))
        return phoneme_tensor, mel_spectrogram

#  Collate Function
def collate_fn(batch):
    phoneme_tensors, spectrograms = zip(*batch)
    phoneme_tensors = pad_sequence(phoneme_tensors, batch_first=True, padding_value=0)
    max_time = max(s.shape[1] for s in spectrograms)
    spectrograms = [torch.nn.functional.pad(s, (0, max_time - s.shape[1]), mode='constant', value=0) for s in spectrograms]
    spectrograms = torch.stack(spectrograms).permute(0, 2, 1)
    return phoneme_tensors, spectrograms

#  Tacotron Model
class Tacotron(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Tacotron, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self._initialize_weights()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)

#  Load Dataset
dataset_path = "C:/Users/DELL/Downloads/ttsnew/LJSpeech-1.1"
metadata_path = os.path.join(dataset_path, "metadata_phonemes.csv")
spectrogram_dir = os.path.join(dataset_path, "melspectrograms")
dataset = LJSpeechDataset(metadata_path, spectrogram_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

#  Initialize Model, Loss, Optimizer, Scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(dataset.phoneme_to_idx) + 1
embedding_dim = 64
hidden_dim = 512  # Updated to match checkpoint
output_dim = 80

model = Tacotron(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

#  Training Loop
def train(model, dataloader, epochs=20, accumulate_steps=2):
    model.train()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()

        for step, (phonemes, spectrograms) in enumerate(dataloader):
            phonemes, spectrograms = phonemes.to(device), spectrograms.to(device)

            # Normalize spectrograms
            spectrograms = spectrograms / torch.max(abs(spectrograms))

            outputs = model(phonemes)
            spectrograms = spectrograms[:, :outputs.shape[1], :]

            if outputs.shape != spectrograms.shape:
                print(f"Warning: Mismatched shapes {outputs.shape} vs {spectrograms.shape}")
                continue

            loss = criterion(outputs, spectrograms) / accumulate_steps
            loss.backward()

            if (step + 1) % accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulate_steps

        scheduler.step(total_loss / len(dataloader))  # Update scheduler per epoch
        print(f" Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    #  Save Model
    torch.save(model.state_dict(), "trained_tacotron.pth")
    print(" Model saved as 'trained_tacotron.pth'")

# Start Training
#train(model, dataloader, epochs=20)
