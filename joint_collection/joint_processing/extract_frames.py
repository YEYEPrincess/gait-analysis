import cv2
import subprocess
import json
import os
from datetime import datetime

def get_video_offset_ms(video_path):
    """Return video start time (ms since epoch) from metadata."""

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "format_tags=creation_time",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    creation_time = data["format"]["tags"]["creation_time"]
    print(data["format"]["tags"])

    dt = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def compute_timestamp_ms(frame_index, fps, offset_ms):
    """Return absolute timestamp (ms) of a frame."""
    return int(offset_ms + (frame_index / fps) * 1000)

def extract_frames_16fps(video_path, output_folder, target_fps=16):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input_videos")
    output_root = os.path.join(script_dir, "output_frames")
    output_dir = os.path.join(output_root, output_folder)

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    full_video_path = os.path.join(input_dir, video_path)
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {full_video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if not original_fps or original_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS ({original_fps}) for video: {full_video_path}")

    step = max(1, round(original_fps / target_fps))

    # offset_ms = get_video_offset_ms(full_video_path)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            # ts = compute_timestamp_ms(frame_count, original_fps, offset_ms)
            ts = int(frame_count/step)  # Use frame index as timestamp for simplicity
            filename = os.path.join(output_dir, f"{ts:08d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Saved {saved_count} frames to {output_dir}")

"""if __name__ == "__main__":
    video_path = "sample_video_1.mov"
    output_folder = video_path.split('.')[0]
    extract_frames_16fps(video_path, output_folder)"""
