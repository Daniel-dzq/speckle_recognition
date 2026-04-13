import subprocess
import os
import glob
import math

# ================== Parameters (edit here) ==================
# Use relative paths (relative to this script's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "screenshots")
TARGET_FRAMES = 100          # Number of frames to extract per video
IMAGE_FORMAT = "png"         # png / jpg
# =======================================================


def get_ffmpeg_path():
    """Prefer imageio-ffmpeg (bundled ffmpeg) when available."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_video_duration(video_path, ffmpeg_path):
    """Get video duration in seconds using ffmpeg."""
    cmd = [ffmpeg_path, "-i", video_path]
    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    for line in result.stderr.splitlines():
        if "Duration" in line:
            # Duration: 00:00:12.34
            time_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = time_str.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)

    return None


def extract_frames(video_path, output_dir, ffmpeg_path):
    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(video_path, ffmpeg_path)
    if not duration or duration <= 0:
        print("  [WARN] Could not get video duration, skipping")
        return 0

    fps = TARGET_FRAMES / duration
    fps = max(fps, 0.1)  # avoid edge cases

    output_pattern = os.path.join(output_dir, f"frame_%05d.{IMAGE_FORMAT}")

    cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-vf", f"fps={fps},format=gray",
        "-vsync", "vfr",
        "-y",
        output_pattern
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        print("  [FAIL] ffmpeg execution failed")
        return 0

    files = glob.glob(os.path.join(output_dir, f"*.{IMAGE_FORMAT}"))
    return len(files)


def main():
    ffmpeg_path = get_ffmpeg_path()
    print(f"Using ffmpeg: {ffmpeg_path}")

    video_files = []
    for ext in ("avi", "AVI", "mp4", "mkv", "mov"):
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, f"*.{ext}")))

    if not video_files:
        print("[FAIL] No video files found")
        return

    print(f"Found {len(video_files)} video(s)")
    print(f"Target: {TARGET_FRAMES} frame(s) per video")
    print("=" * 60)

    total = 0

    for video in video_files:
        name = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(OUTPUT_DIR, name)

        print(f"\nProcessing: {name}")
        saved = extract_frames(video, out_dir, ffmpeg_path)
        print(f"  [OK] Saved {saved} frame(s)")
        total += saved

    print("=" * 60)
    print(f"Done. Saved {total} image(s) total.")


if __name__ == "__main__":
    main()
