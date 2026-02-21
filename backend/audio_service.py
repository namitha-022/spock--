import torch
import librosa
import numpy as np
import subprocess
import os

MODEL_PATH = "crnn_audio_fake.pth"
SR = 16000
MAX_LEN = SR * 8

# --------------------
# Load model
# --------------------
class CRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.rnn = torch.nn.GRU(32*32, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, -1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return self.fc(x)

model = CRNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --------------------
# Utilities
# --------------------
def fix_length(audio):
    if len(audio) > MAX_LEN:
        return audio[:MAX_LEN]
    return np.pad(audio, (0, MAX_LEN - len(audio)))

def wav_to_mel(path):
    audio, _ = librosa.load(path, sr=SR)
    audio = fix_length(audio)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=128, n_fft=1024, hop_length=512
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.nan_to_num(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

# --------------------
# Extract audio from video
# --------------------
def extract_audio(video_path, out_path="temp.wav"):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

# --------------------
# Main inference
# --------------------
def analyze_audio(video_path):
    wav_path = extract_audio(video_path)

    mel = wav_to_mel(wav_path)

    with torch.no_grad():
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)

    fake_prob = probs[0][1].item()

    os.remove(wav_path)

    return {
        "fake_probability": fake_prob,
        "label": "Fake" if fake_prob > 0.5 else "Real"
    }
