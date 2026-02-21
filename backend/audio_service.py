import subprocess

def has_audio(video_path: str) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            video_path
        ],
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())


def analyze_audio(video_path: str):
    if not has_audio(video_path):
        return {
            "audio_score": None,
            "status": "skipped",
            "reason": "No audio stream in video"
        }

    # 🔽 audio processing goes here (only if audio exists)
    return {
        "audio_score": 0.87,
        "status": "processed"
    }