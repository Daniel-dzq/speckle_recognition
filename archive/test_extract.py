#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test video frame extraction"""

import subprocess
import os

def test_extract():
    # Resolve FFmpeg path
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"FFmpeg: {ffmpeg}")
    except:
        ffmpeg = "ffmpeg"

    # Video path
    video = input("Enter full path to video file: ").strip().strip('"')

    if not os.path.exists(video):
        print(f"File not found: {video}")
        return

    # Output directory
    output_dir = os.path.join(os.path.dirname(video), "test_output")
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "frame_%05d.png")

    # Extraction command
    cmd = [
        ffmpeg,
        "-i", video,
        "-vf", "fps=5,format=gray",
        "-vsync", "vfr",
        "-y",
        output_pattern
    ]

    print(f"\nRunning command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()

    # Run
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="ignore"
    )

    print(f"Return code: {result.returncode}")

    if result.stderr:
        print(f"\nStderr (last 20 lines):")
        lines = result.stderr.strip().split('\n')
        for line in lines[-20:]:
            print(f"  {line}")

    # Check output files
    import glob
    files = glob.glob(os.path.join(output_dir, "*.png"))
    print(f"\nFiles created: {len(files)}")

    if files:
        print("Success. Files:")
        for f in files[:5]:
            print(f"  {os.path.basename(f)}")
        if len(files) > 5:
            print(f"  ... {len(files)} files total")
    else:
        print("Failed: no output files were created")

if __name__ == "__main__":
    test_extract()
    input("\nPress Enter to exit...")
