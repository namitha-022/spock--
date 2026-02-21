from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio_service import analyze_audio

audio_path = str(Path(__file__).resolve().parents[1] / "media" / "sample4.mp4")
result = analyze_audio(audio_path)
print(result)
