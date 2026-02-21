from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.video_service import task_video_analysis
from backend.audio_service import task_audio_analysis

sample_path = str((Path(__file__).resolve().parent / "media" / "real_sample1.mp4"))

video_task = task_video_analysis.delay(sample_path)
audio_task = task_audio_analysis.delay(sample_path)

print("Video:", video_task.get())
print("Audio:", audio_task.get())
