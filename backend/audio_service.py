import subprocess
import librosa
import numpy as np
import torch
import tempfile
import os

# ------------------
# CONSTANTS
# ------------------
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
MAX_AUDIO_SECONDS = 8
MAX_LEN = SAMPLE_RATE * MAX_AUDIO_SECONDS

torch.set_num_threads(2)

# ------------------
# SAFE MP4 → WAV
# ------------------
def extract_wav(video_path: str, wav_path: str) -> bool:
    """
    Extract audio if present.
    Works even if stream index is missing.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-i", video_path,
                "-map", "0:a?",          # 🔥 SAFE MAP
                "-vn",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                wav_path
            ],
            check=True
        )

        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 1024

    except subprocess.CalledProcessError:
        return False


# ------------------
# WAV → MEL
# ------------------
def wav_to_mel(wav_path: str):
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.nan_to_num(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = librosa.util.fix_length(mel, size=128, axis=1)

    return torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()


# ------------------
# MAIN ENTRY
# ------------------
def analyze_audio(video_path: str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        ok = extract_wav(video_path, tmp_wav.name)

        if not ok:
            return {
                "audio_score": None,
                "status": "skipped",
                "reason": "No decodable audio stream"
            }

        mel = wav_to_mel(tmp_wav.name)

    return {
        "audio_score": None,
        "status": "processed",
        "mel_shape": list(mel.shape)
    }