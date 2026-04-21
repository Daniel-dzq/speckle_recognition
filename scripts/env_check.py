#!/usr/bin/env python3
"""
Environment check script for Speckle-PUF training.
Prints Python, PyTorch, CUDA, and GPU diagnostics.

Usage:
    python scripts/env_check.py
"""

import sys
import os
import platform

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

LINE = "=" * 60


def check_python():
    print(f"\n{'Python':>20}: {sys.version}")
    print(f"{'Platform':>20}: {platform.platform()}")


def check_torch():
    try:
        import torch
        print(f"\n{'PyTorch version':>20}: {torch.__version__}")

        try:
            import torchvision
            print(f"{'torchvision':>20}: {torchvision.__version__}")
        except ImportError:
            print(f"{'torchvision':>20}: NOT INSTALLED")

        print(f"{'CUDA available':>20}: {torch.cuda.is_available()}")
        print(f"{'torch.version.cuda':>20}: {torch.version.cuda}")
        print(f"{'cuDNN available':>20}: {torch.backends.cudnn.is_available()}")
        print(f"{'cuDNN version':>20}: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f"{'Device count':>20}: {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024 ** 3
                print(f"{'GPU ' + str(i):>20}: {props.name}  ({mem_gb:.1f} GB)")

            print(f"\n  Running GPU compute test ...")
            t = torch.zeros(1024, 1024, device="cuda")
            r = t @ t.T
            del t, r
            print(f"  GPU compute test        : PASSED")
            print(f"\n  [OK] CUDA is ready for training.")
        else:
            print(f"\n  [WARNING] CUDA not available. Training will run on CPU.")
            print()
            print("  To enable GPU (RTX 5080, CUDA 12.x), run:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
            print("  Or for nightly (sm_120 blackwell):")
            print("    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")

    except ImportError:
        print("\n  [ERROR] PyTorch is NOT installed.")
        print("  Install command:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")


def check_cv2():
    try:
        import cv2
        print(f"\n{'OpenCV':>20}: {cv2.__version__}")
    except ImportError:
        print(f"\n{'OpenCV':>20}: NOT INSTALLED  (pip install opencv-python)")


def check_sklearn():
    try:
        import sklearn
        print(f"{'scikit-learn':>20}: {sklearn.__version__}")
    except ImportError:
        print(f"{'scikit-learn':>20}: NOT INSTALLED  (pip install scikit-learn)")


def check_other():
    pkgs = ["tqdm", "matplotlib", "numpy", "PIL", "PySide6"]
    for pkg in pkgs:
        try:
            m = __import__(pkg if pkg != "PIL" else "PIL.Image", fromlist=[""])
            ver = getattr(m, "__version__", "ok")
            # PySide6 version
            if pkg == "PySide6":
                import PySide6
                ver = PySide6.__version__
            print(f"{pkg:>20}: {ver}")
        except ImportError:
            print(f"{pkg:>20}: NOT INSTALLED")


def check_mvsdk():
    """Check whether the MindVision CCD SDK library can be loaded."""
    import platform as _plat
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        import gui.mvsdk as mvsdk
        mvsdk.get_lib()
        mvsdk.sdk_init(0)
        devs = mvsdk.enumerate_devices()
        n = len(devs)
        names = [d.friendly_name or d.product_name for d in devs]
        cam_info = f"{n} camera(s) detected" + (f": {names}" if names else "")
        print(f"\n{'MindVision SDK':>20}: OK  —  {cam_info}")
    except FileNotFoundError:
        arch = _plat.architecture()[0]
        if sys.platform == "darwin":
            lib = "libmvsdk.dylib"
            hint = "copy to gui/  then  codesign --force --sign - gui/libmvsdk.dylib"
        elif sys.platform == "win32":
            lib = "MVCAMSDK_X64.dll" if arch == "64bit" else "MVCAMSDK.dll"
            hint = f"copy {lib} to gui/  (install MindVision Windows SDK first)"
        else:
            lib = "libMVSDK.so"
            hint = f"copy {lib} to gui/"
        print(f"\n{'MindVision SDK':>20}: NOT FOUND  ({lib})")
        print(f"  Hint: {hint}")
    except OSError as e:
        print(f"\n{'MindVision SDK':>20}: LOAD ERROR — {e}")
    except Exception as e:
        print(f"\n{'MindVision SDK':>20}: ERROR — {e}")


def check_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dir = os.path.join(base, "video_capture")
    print(f"\n{'Data directory':>20}: {video_dir}")
    if not os.path.isdir(video_dir):
        print("  [WARNING] video_capture/ directory not found")
        return
    fibers = [d for d in sorted(os.listdir(video_dir)) if os.path.isdir(os.path.join(video_dir, d))]
    print(f"{'Fibers found':>20}: {fibers}")
    for fiber in fibers:
        fiber_dir = os.path.join(video_dir, fiber)
        avis = [f for f in os.listdir(fiber_dir) if f.lower().endswith(".avi")]
        print(f"  {fiber:<18}: {len(avis)} .avi files")


def main():
    print(LINE)
    print("  Speckle-PUF Environment Check")
    print(LINE)
    check_python()
    check_torch()
    check_cv2()
    check_sklearn()
    check_other()
    check_mvsdk()
    check_data()
    print(f"\n{LINE}")
    print("  Check complete.")
    print(LINE)


if __name__ == "__main__":
    main()
