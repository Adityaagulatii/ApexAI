import sys
import os
import shutil
import subprocess

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "frames")
VIDEO_OUT = os.path.join(os.path.dirname(__file__), "video.mp4")


def ingest(video_path: str) -> None:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[ingest] Copying {video_path} -> {VIDEO_OUT}")
    shutil.copy2(video_path, VIDEO_OUT)

    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    print("[ingest] Extracting frames at 10fps...")
    cmd = [
        "ffmpeg", "-y",
        "-i", VIDEO_OUT,
        "-vf", "fps=10",
        os.path.join(FRAMES_DIR, "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    frame_count = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    print(f"[ingest] Done. {frame_count} frames extracted to {FRAMES_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <video_path>")
        sys.exit(1)
    ingest(sys.argv[1])
