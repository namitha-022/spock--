def compute_final_score(video_result, audio_result, metadata_result):

    v = video_result.get("video_score", 0.5)
    a = audio_result.get("audio_probability", 0.5)
    m = metadata_result.get("metadata_score", 0.0)  # default clean

    # Clamp values for safety (avoid weird >1 or <0 cases)
    v = max(0.0, min(1.0, v))
    a = max(0.0, min(1.0, a))
    m = max(0.0, min(1.0, m))

    # Probabilistic OR fusion
    final_score = 1 - (
    (1 - v) *
    (1 - (a * 0.6)) *
    (1 - m * 0.8)
)

    # Strong override if recycled
    if metadata_result.get("recycled"):
        final_score = max(final_score, 0.9)

    verdict = "Likely Fake" if final_score > 0.6 else "Likely Real"

    return {
        "final_score": round(final_score, 3),
        "verdict": verdict
    }