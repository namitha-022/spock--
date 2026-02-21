import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# -------------------------
# CPU SAFETY
# -------------------------
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
DEVICE = "cpu"

DATASET_PATH = "../audio-dataset"
SAMPLE_RATE = 16000
DURATION = 8
MAX_LEN = SAMPLE_RATE * DURATION

# -------------------------
# LOAD DATA
# -------------------------
def load_data():
    X, y = [], []

    for label, folder in enumerate(["real", "fake"]):
        path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(path):
            print(f"[WARN] Missing folder: {path}")
            continue

        for file in os.listdir(path):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(path, file)

            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                if audio is None or len(audio) < 1600:   # <0.1 sec @16kHz
                    continue

                # FIX LENGTH
                if len(audio) > MAX_LEN:
                    audio = audio[:MAX_LEN]
                else:
                    audio = np.pad(audio, (0, MAX_LEN - len(audio)))

                # MEL
                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=128,
                    n_fft=1024,
                    hop_length=512
                )
                mel = librosa.power_to_db(mel, ref=np.max)

                # NORMALIZE + NaN SAFE
                mel = np.nan_to_num(mel)
                mel = (mel - mel.mean()) / (mel.std() + 1e-6)

                X.append(mel)
                y.append(label)

            except Exception as e:
                print(f"[SKIP] {file_path} → {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Loaded samples: {len(X)}")
    return X, y

# -------------------------
# CRNN MODEL
# -------------------------
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(
            input_size=32 * 32,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return torch.sigmoid(x)

# -------------------------
# TRAIN
# -------------------------
def train():
    X, y = load_data()
    assert len(X) > 0, "Dataset is empty — check audio-dataset path"

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train).unsqueeze(1).float()
    y_train = torch.tensor(y_train).float().unsqueeze(1)

    model = CRNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/5 | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "crnn_audio.pth")
    print("✅ Saved crnn_audio.pth")

if __name__ == "__main__":
    train()