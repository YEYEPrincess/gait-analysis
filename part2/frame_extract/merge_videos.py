import os
import cv2
import math
import numpy as np
import subprocess

base_dir = os.path.dirname(os.path.abspath(__file__))

skeleton_video_path = os.path.join(base_dir, "skeleton_animation.mp4")
sample_frames_dir = os.path.join(base_dir, "output_frames", "sample_video")

temp_output_path = os.path.join(base_dir, "combined_temp.avi")
final_output_path = os.path.join(base_dir, "combined.mp4")

speed_factor = 0.5
fps_out = 16
repeat_count = int(round(1 / speed_factor)) if speed_factor < 1 else 1

image_extensions = {".png", ".jpg", ".jpeg"}

def numeric_stem_key(filename: str):
    stem = os.path.splitext(filename)[0]
    try:
        return int(stem)
    except ValueError:
        return stem

def get_center_y_offset(video_h: int, canvas_h: int) -> int:
    return (canvas_h - video_h) // 2

frame_files = [
    f for f in os.listdir(sample_frames_dir)
    if os.path.splitext(f)[1].lower() in image_extensions
]
frame_files = sorted(frame_files, key=numeric_stem_key)

if not frame_files:
    raise ValueError(f"No image frames found in {sample_frames_dir}")

cap_skel = cv2.VideoCapture(skeleton_video_path)

if not cap_skel.isOpened():
    raise ValueError(f"Cannot open {skeleton_video_path}")

w1 = int(cap_skel.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap_skel.get(cv2.CAP_PROP_FRAME_HEIGHT))

first_frame_path = os.path.join(sample_frames_dir, frame_files[0])
first_img = cv2.imread(first_frame_path)

if first_img is None:
    raise ValueError(f"Cannot read first frame: {first_frame_path}")

h2, w2 = first_img.shape[:2]

out_w = w2 + w1
out_h = max(h2, h1)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(temp_output_path, fourcc, fps_out, (out_w, out_h))

if not writer.isOpened():
    raise ValueError("Could not open VideoWriter")

y_skel = get_center_y_offset(h1, out_h)
y_real = get_center_y_offset(h2, out_h)

frame_index = 0

while True:
    if frame_index >= len(frame_files):
        break

    ret_skel, skeleton_frame = cap_skel.read()
    if not ret_skel:
        break

    real_frame_path = os.path.join(sample_frames_dir, frame_files[frame_index])
    real_frame = cv2.imread(real_frame_path)

    if real_frame is None:
        print(f"Skipping unreadable frame: {real_frame_path}")
        frame_index += 1
        continue

    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    # Real frame on the left
    canvas[y_real:y_real + h2, 0:w2] = real_frame

    # Skeleton video on the right
    canvas[y_skel:y_skel + h1, w2:w2 + w1] = skeleton_frame

    for _ in range(repeat_count):
        writer.write(canvas)

    frame_index += 1

cap_skel.release()
writer.release()

subprocess.run([
    "ffmpeg",
    "-y",
    "-i", temp_output_path,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    final_output_path
], check=True)

os.remove(temp_output_path)

print(f"Saved to {final_output_path}")
print(f"Used {frame_index} synchronized frame pairs")
