#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FFmpeg diagnostic script"""

import subprocess
import os
import glob

def find_ffmpeg():
    """Locate FFmpeg executable."""
    # 1. Try imageio_ffmpeg
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[OK] imageio_ffmpeg found: {path}")
        return path
    except Exception as e:
        print(f"[FAIL] imageio_ffmpeg not available: {e}")

    # 2. Try system PATH
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        if result.returncode == 0:
            print("[OK] ffmpeg found on system PATH")
            return "ffmpeg"
        else:
            print(f"[FAIL] system ffmpeg returned error: {result.returncode}")
    except Exception as e:
        print(f"[FAIL] system ffmpeg not available: {e}")

    # 3. Common install paths
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            print(f"[OK] FFmpeg found: {path}")
            return path
        else:
            print(f"[FAIL] does not exist: {path}")

    return None

def test_video(ffmpeg_path, video_path):
    """Probe a video file with ffmpeg."""
    print(f"\nTesting video: {video_path}")

    if not os.path.exists(video_path):
        print(f"  [FAIL] file does not exist!")
        return

    print(f"  File size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")

    # Get video metadata
    cmd = [ffmpeg_path, "-i", video_path]
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    print(f"  Return code: {result.returncode}")

    # Look for Duration
    duration_found = False
    for line in result.stderr.splitlines():
        if "Duration" in line:
            print(f"  Duration line: {line.strip()}")
            duration_found = True
        if "Stream" in line and "Video" in line:
            print(f"  Video stream: {line.strip()}")

    if not duration_found:
        print("  [FAIL] no duration line found!")
        print("  FFmpeg output:")
        for line in result.stderr.splitlines()[:10]:
            print(f"    {line}")

def main():
    print("=" * 50)
    print("FFmpeg diagnostic tool")
    print("=" * 50)

    ffmpeg_path = find_ffmpeg()

    if not ffmpeg_path:
        print("\n[ERROR] FFmpeg not found!")
        print("Install FFmpeg:")
        print("  1. Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip")
        print("  2. Extract to C:\\ffmpeg")
        print("  3. Ensure C:\\ffmpeg\\bin\\ffmpeg.exe exists")
        return

    print(f"\nTesting FFmpeg version...")
    result = subprocess.run(
        [ffmpeg_path, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print(f"  {result.stdout.splitlines()[0]}")

    print("\n" + "=" * 50)
    print("Looking for video files...")

    video_dir = input("Enter path to folder containing videos: ").strip().strip('"')

    if not os.path.isdir(video_dir):
        print(f"[FAIL] folder does not exist: {video_dir}")
        return

    videos = []
    for ext in ['*.avi', '*.mp4', '*.mkv', '*.mov', '*.AVI', '*.MP4']:
        videos.extend(glob.glob(os.path.join(video_dir, ext)))

    print(f"Found {len(videos)} video file(s)")

    for video in videos[:3]:
        test_video(ffmpeg_path, video)

    print("\n" + "=" * 50)
    print("Diagnostic complete")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
