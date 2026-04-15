import os
import shutil

from extract_frames import extract_frames_16fps

FRAME_EXTENSION = ".jpg"

def numeric_frame_sort_key(filename: str) -> int:
    name, ext = os.path.splitext(filename)
    if ext.lower() != FRAME_EXTENSION:
        return float("inf")
    return int(name)


def calibrate_extracted_frames(output_dir: str, start_frame_number: int, target_fps: int = 16) -> None:
    frame_files = [f for f in os.listdir(output_dir) if f.lower().endswith(FRAME_EXTENSION)]
    frame_files = sorted(frame_files, key=numeric_frame_sort_key)

    if not frame_files:
        raise ValueError(f"No {FRAME_EXTENSION.upper()} files found in {output_dir}")

    frame_numbers = [numeric_frame_sort_key(f) for f in frame_files]

    if start_frame_number not in frame_numbers:
        raise ValueError(
            f"Selected start frame {start_frame_number} not found in extracted frames.\n"
            f"Available range: {frame_numbers[0]} to {frame_numbers[-1]}"
        )

    selected_files = [f for f in frame_files if numeric_frame_sort_key(f) >= start_frame_number]

    ms_per_frame = 1000.0 / target_fps

    temp_dir = os.path.join(output_dir, "__tmp_calibrated__")
    os.makedirs(temp_dir, exist_ok=True)

    for f in selected_files:
        old_idx = numeric_frame_sort_key(f)
        new_idx = old_idx - start_frame_number
        timestamp_ms = int(round(new_idx * ms_per_frame))

        src = os.path.join(output_dir, f)
        dst = os.path.join(temp_dir, f"{timestamp_ms}{FRAME_EXTENSION}")
        shutil.copy2(src, dst)

    for f in frame_files:
        os.remove(os.path.join(output_dir, f))

    for f in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, f), os.path.join(output_dir, f))

    os.rmdir(temp_dir)

    print(f"Calibration done in: {output_dir}")
    print(f"First kept frame: {start_frame_number} -> 0{FRAME_EXTENSION}")


def main():
    video_path = "sample_video_1.mov"
    target_fps = 16

    output_folder = os.path.splitext(video_path)[0]

    extract_frames_16fps(video_path, output_folder, target_fps=target_fps)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output_frames", output_folder)

    start_frame_number = int(
        input("Enter the start frame number chosen by manual inspection: ").strip()
    )

    calibrate_extracted_frames(
        output_dir=output_dir,
        start_frame_number=start_frame_number,
        target_fps=target_fps
    )


if __name__ == "__main__":
    main()
