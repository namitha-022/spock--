import os
import json
import hashlib
import subprocess
from pathlib import Path

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_meta"
HASH_DB = BASE_DIR / "hash_db.json"

os.makedirs(TEMP_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1️⃣ SHA-256 (Exact Duplicate Detection)
# ------------------------------------------------------------------

def compute_sha256(video_path):
    sha256 = hashlib.sha256()

    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ------------------------------------------------------------------
# 2️⃣ Extract One Frame Using FFmpeg
# ------------------------------------------------------------------

def extract_frame(video_path):
    frame_path = TEMP_DIR / "frame.jpg"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-frames:v", "1",
        str(frame_path),
        "-y"
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return str(frame_path)


# ------------------------------------------------------------------
# 3️⃣ Perceptual Hash (pHash)
# ------------------------------------------------------------------

def compute_phash(image_path):
    from PIL import Image
    import imagehash

    img = Image.open(image_path)
    phash_value = str(imagehash.phash(img))
    return phash_value


# ------------------------------------------------------------------
# 4️⃣ Check Recycled Media (Hamming Distance Matching)
# ------------------------------------------------------------------

def check_recycled(phash_value):
    import imagehash

    if not HASH_DB.exists():
        return False

    with HASH_DB.open("r") as f:
        db = json.load(f)

    current_hash = imagehash.hex_to_hash(phash_value)

    for stored_hash in db:
        stored_hash_obj = imagehash.hex_to_hash(stored_hash)
        distance = current_hash - stored_hash_obj

        if distance < 5:  # threshold
            return True

    return False


def store_hash(phash_value):
    if HASH_DB.exists():
        with HASH_DB.open("r") as f:
            db = json.load(f)
    else:
        db = []

    db.append(phash_value)

    with HASH_DB.open("w") as f:
        json.dump(db, f)


# ------------------------------------------------------------------
# 5️⃣ Basic FFmpeg Metadata Extraction (Optional but Nice)
# ------------------------------------------------------------------

def extract_basic_metadata(video_path):
    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout


# ------------------------------------------------------------------
# 6️⃣ MAIN METADATA ANALYZER
# ------------------------------------------------------------------

def analyze_metadata(video_path):
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        return {
            "type": "metadata",
            "metadata_score": 0.0,
            "recycled": False,
            "error": f"Missing dependency: {e}. Install Pillow and ImageHash.",
        }


    resolved_video_path = Path(video_path)
    if not resolved_video_path.exists():
        candidate = BASE_DIR / video_path
        if candidate.exists():
            resolved_video_path = candidate
        else:
            return {
                "type": "metadata",
                "metadata_score": 0.0,
                "recycled": False,
                "error": f"File not found: {video_path}",
            }

    sha = compute_sha256(str(resolved_video_path))

    frame = extract_frame(str(resolved_video_path))

    phash_value = compute_phash(frame)

    recycled = check_recycled(phash_value)

    # scoring logic
    score = 0.0

    if recycled:
      score = 0.85
    else:
      metadata_json = extract_basic_metadata(str(resolved_video_path))

      if "Lavf" in metadata_json:
        score += 0.1

      if "encoder" not in metadata_json:
        score += 0.1

      score = min(score, 0.4)

    metadata_json = extract_basic_metadata(str(resolved_video_path))

    # cleanup frame
    if os.path.exists(frame):
        os.remove(frame)

    result = {
        "type": "metadata",
        "metadata_score": round(score, 2),
        "recycled": recycled,
        "details": {
            "sha256": sha,
            "phash": phash_value,
        }
    }

    return result

from backend.celery_app import celery

@celery.task(name="metadata_service.task_metadata_analysis")
def task_metadata_analysis(video_path):
    return analyze_metadata(video_path)
