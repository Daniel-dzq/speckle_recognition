#!/usr/bin/env python3
"""
Automated Qt grab() screenshots of MainWindow, SLMWindow, and RobotPanel.

  python scripts/capture_manual_screenshots.py --auto --native

Environment variables (only --auto sets these in-process):

  SPECKLE_MANUAL_SCREENSHOT_MODE=1  — enables manual-screenshot-only branches in the GUI
  SPECKLE_FORCE_AUTH_STATE          — granted | denied_class for RobotPanel staging
                                      (see gui/robot_panel.py)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI_DIR = os.path.join(ROOT, "gui")

if sys.platform == "win32":
    for _d in [
        os.path.join(GUI_DIR, "win_sdk"),
        GUI_DIR,
        r"C:\Program Files\MindVision\SDK",
        r"C:\Program Files (x86)\MindVision\SDK",
        r"C:\MindVision\SDK",
    ]:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except AttributeError:
                os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")


def _pulse(app, n: int = 80) -> None:
    from PySide6.QtCore import QCoreApplication

    for _ in range(n):
        app.processEvents()
        QCoreApplication.processEvents()


def _save_pm(pm, path: str) -> bool:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return bool(pm.save(path, "PNG"))


def grab_combo_horizontal(main_win, path: str, gap: int = 12) -> bool:
    from PySide6.QtGui import QImage, QPainter, QPixmap

    slm = getattr(main_win, "_slm_window", None)
    if slm is None or not slm.isVisible():
        print("[auto] SLM window missing; skip composite", file=sys.stderr)
        return False
    main_win.raise_()
    main_win.activateWindow()
    slm.raise_()
    pm_m = main_win.grab()
    pm_s = slm.grab()
    w = pm_m.width() + gap + pm_s.width()
    h = max(pm_m.height(), pm_s.height())
    canvas = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(0xFF2A2A3C)
    painter = QPainter(canvas)
    painter.drawPixmap(0, (h - pm_m.height()) // 2, pm_m)
    painter.drawPixmap(pm_m.width() + gap, (h - pm_s.height()) // 2, pm_s)
    painter.end()
    return _save_pm(QPixmap.fromImage(canvas), path)


def ensure_sample_video(video_dir: str) -> str | None:
    os.makedirs(video_dir, exist_ok=True)
    dest = os.path.join(video_dir, "manual_capture_preview.avi")
    if os.path.isfile(dest) and os.path.getsize(dest) > 512:
        return dest
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    try:
        subprocess.run(
            [
                ff,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=640x480:d=1",
                "-pix_fmt",
                "yuv420p",
                dest,
            ],
            check=True,
            timeout=90,
            capture_output=True,
        )
        return dest if os.path.isfile(dest) else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def print_help_manual(out_dir: str) -> None:
    od = os.path.abspath(out_dir)
    lines = [
        "=" * 66,
        "  Manual capture checklist (fixed filenames under output directory)",
        "=" * 66,
        f"Repository: {ROOT}",
        f"Output dir: {od}",
        "",
        "Recommended:  python scripts/capture_manual_screenshots.py --auto --native",
        "",
        "Fig 3-1  screenshot_step_home.png",
        "Fig 3-2  screenshot_camera_permission.png  (use a real OS permission UI if required)",
        "Fig 4-3-1 … 4-3-10  same basename order as the --auto sequence below",
        "=" * 66,
    ]
    print("\n".join(lines))


def run_auto(out_dir: str) -> list[dict]:
    """Drive the live UI; returns rows for the summary table."""
    os.environ["SPECKLE_MANUAL_SCREENSHOT_MODE"] = "1"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.pop("SPECKLE_FORCE_AUTH_STATE", None)
    sys.path.insert(0, ROOT)

    from PySide6.QtWidgets import QApplication

    from gui.main_window import FIBER_MODELS_DIR, MainWindow, discover_fiber_models

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ensure_sample_video(os.path.join(ROOT, "video_capture"))

    rows: list[dict] = []

    app = QApplication(["speckle_auto_capture"])
    app.setStyle("Fusion")

    win = MainWindow()
    win.resize(1600, 980)
    win.show()
    win.raise_()
    win.activateWindow()
    _pulse(app, 120)

    def snap(path: str, note_ok: str, note_fail: str = "") -> bool:
        ok = _save_pm(win.grab(), path)
        rows.append(
            {
                "file": os.path.basename(path),
                "status": "ok" if ok else "failed",
                "note": note_ok if ok else note_fail,
            }
        )
        return ok

    # 1 home
    p = os.path.join(out_dir, "screenshot_step_home.png")
    snap(p, "main window default")

    # 2 permission (never fake OS dialog)
    perm_path = os.path.join(out_dir, "screenshot_camera_permission.png")
    if os.path.isfile(perm_path) and os.path.getsize(perm_path) > 64:
        rows.append(
            {
                "file": "screenshot_camera_permission.png",
                "status": "kept",
                "note": "existing file left in place (assumed hand-captured OS permission UI)",
            }
        )
        print(
            "\n[Fig 3-2] screenshot_camera_permission.png already exists; not overwritten.\n"
            "  Replace manually with a real OS camera-permission dialog if you need one.\n"
        )
    else:
        g = getattr(win, "_group_camera_video", None)
        if g is not None:
            _save_pm(g.grab(), perm_path)
            rows.append(
                {
                    "file": "screenshot_camera_permission.png",
                    "status": "ok_placeholder",
                    "note": "Camera / Video group grab; not the OS dialog",
                }
            )
        print(
            "[Fig 3-2] saved Camera / Video panel as a layout placeholder.\n"
            "  For publication, capture macOS / Windows Settings → Privacy & Security → Camera "
            "or the system permission sheet and overwrite:\n"
            f"  {perm_path}\n"
        )

    # 3 scan (indices 0–1 typical in manual mode)
    win._scan_cameras()  # pylint: disable=protected-access
    _pulse(app, 180)
    snap(os.path.join(out_dir, "screenshot_step_scan_cameras.png"), "after Scan (0–1)")

    # 4 manual video feed
    win.start_manual_screenshot_feed()  # pylint: disable=protected-access
    _pulse(app, 220)
    snap(os.path.join(out_dir, "screenshot_step_camera_on.png"), "manual speckle placeholder feed")

    # 5 model (MainWindow auto-loads at startup; use placeholder only if weights missing / load failed)
    models = discover_fiber_models(FIBER_MODELS_DIR)
    _pulse(app, 500)
    if models:
        worker = win._infer_worker  # pylint: disable=protected-access
        loaded = getattr(worker, "_model", None) is not None
        if not loaded:
            win.apply_manual_screenshot_model_fallback_ui()  # pylint: disable=protected-access
            print(
                "[auto] inference worker has no model after startup; using placeholder UI",
                file=sys.stderr,
            )
    else:
        win.apply_manual_screenshot_model_fallback_ui()  # pylint: disable=protected-access
        print("[auto] no Fiber*.pth found; using placeholder Loaded UI", file=sys.stderr)
    _pulse(app, 200)
    snap(os.path.join(out_dir, "screenshot_step_model_loaded.png"), "Loaded / placeholder")

    # 6 recognition off
    win._chk_infer_active.setChecked(False)  # pylint: disable=protected-access
    _pulse(app, 200)
    snap(os.path.join(out_dir, "screenshot_step_recognition_off.png"), "preview only")

    # 7 reading
    win._chk_infer_active.setChecked(True)  # pylint: disable=protected-access
    win._robot_panel.manual_screenshot_seed_reading_display("Fiber1")  # pylint: disable=protected-access
    _pulse(app, 350)
    snap(os.path.join(out_dir, "screenshot_step_reading_puf.png"), "READING + Top-K")

    # 8 granted
    os.environ["SPECKLE_FORCE_AUTH_STATE"] = "granted"
    win._on_prediction({"top1": "-", "confidence": 0.0, "smoothed": "-", "topk": []})  # pylint: disable=protected-access
    _pulse(app, 650)
    snap(os.path.join(out_dir, "screenshot_step_access_granted.png"), "ACCESS GRANTED")

    # 9 denied
    os.environ["SPECKLE_FORCE_AUTH_STATE"] = "denied_class"
    win._on_prediction({"top1": "-", "confidence": 0.0, "smoothed": "-", "topk": []})  # pylint: disable=protected-access
    _pulse(app, 650)
    snap(os.path.join(out_dir, "screenshot_step_access_denied.png"), "ACCESS DENIED")

    os.environ.pop("SPECKLE_FORCE_AUTH_STATE", None)

    # 10 SLM
    win._robot_panel.on_idle()  # pylint: disable=protected-access
    win._chk_slm_fullscreen.setChecked(False)  # pylint: disable=protected-access
    win._input_letter.setText("A")  # pylint: disable=protected-access
    win._show_slm_on_selected_screen(force_show=True)  # pylint: disable=protected-access
    win._send_to_slm()  # pylint: disable=protected-access
    _pulse(app, 200)
    if win._slm_window is not None:  # pylint: disable=protected-access
        geo = win.geometry()
        sw = win._slm_window  # pylint: disable=protected-access
        sw.move(geo.x() + geo.width() + 24, geo.y() + 40)
        sw.raise_()
    _pulse(app, 400)
    slm_path = os.path.join(out_dir, "screenshot_slm_window.png")
    if not grab_combo_horizontal(win, slm_path) and win._slm_window is not None:  # pylint: disable=protected-access
        _save_pm(win._slm_window.grab(), slm_path)  # pylint: disable=protected-access
    rows.append(
        {
            "file": "screenshot_slm_window.png",
            "status": "ok",
            "note": "main + SLM composite (or SLM only)",
        }
    )

    # 11 file source (no dialog)
    win.apply_manual_screenshot_file_source_ui("manual_demo_video.mp4")  # pylint: disable=protected-access
    _pulse(app, 200)
    snap(
        os.path.join(out_dir, "screenshot_dialog_load_video.png"),
        "Source: File (no file dialog shown)",
    )

    # 12 stop
    win._stop_camera()  # pylint: disable=protected-access
    _pulse(app, 250)
    snap(os.path.join(out_dir, "screenshot_step_stop_camera.png"), "after Stop")

    win.close()
    _pulse(app, 60)
    return rows


def _list_png(out_dir: str) -> list[str]:
    if not os.path.isdir(out_dir):
        return []
    return sorted(f for f in os.listdir(out_dir) if f.lower().endswith(".png"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Grab PNG screenshots from the live demo UI.")
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "figures", "softcopyright"))
    ap.add_argument("--native", action="store_true", help="Use the native platform plugin (required with --auto)")
    ap.add_argument("--auto", action="store_true", help="Fully automated capture (sets manual screenshot mode)")
    ap.add_argument("--help-manual", action="store_true", dest="help_manual")
    ap.add_argument("--wizard", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.help_manual:
        print_help_manual(args.output_dir)
        return 0

    if args.wizard:
        print("Deprecated: use  python scripts/capture_manual_screenshots.py --auto --native", file=sys.stderr)
        return 2

    if args.auto:
        os.environ.pop("QT_QPA_PLATFORM", None)
        if not args.native:
            print("Add --auto --native (needs a real display server).", file=sys.stderr)
            return 2
        rows: list[dict] = []
        try:
            rows = run_auto(args.output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] error: {exc}", file=sys.stderr)
        expected = [
            ("screenshot_step_home.png", "3-1"),
            ("screenshot_camera_permission.png", "3-2"),
            ("screenshot_step_scan_cameras.png", "4-3-1"),
            ("screenshot_step_camera_on.png", "4-3-2"),
            ("screenshot_step_model_loaded.png", "4-3-3"),
            ("screenshot_step_recognition_off.png", "4-3-4"),
            ("screenshot_step_reading_puf.png", "4-3-5"),
            ("screenshot_step_access_granted.png", "4-3-6"),
            ("screenshot_step_access_denied.png", "4-3-7"),
            ("screenshot_slm_window.png", "4-3-8"),
            ("screenshot_dialog_load_video.png", "4-3-9"),
            ("screenshot_step_stop_camera.png", "4-3-10"),
        ]
        legend = {
            "screenshot_step_home.png": "Fig 3-1",
            "screenshot_camera_permission.png": "Fig 3-2",
            "screenshot_step_scan_cameras.png": "Fig 4-3-1",
            "screenshot_step_camera_on.png": "Fig 4-3-2",
            "screenshot_step_model_loaded.png": "Fig 4-3-3",
            "screenshot_step_recognition_off.png": "Fig 4-3-4",
            "screenshot_step_reading_puf.png": "Fig 4-3-5",
            "screenshot_step_access_granted.png": "Fig 4-3-6",
            "screenshot_step_access_denied.png": "Fig 4-3-7",
            "screenshot_slm_window.png": "Fig 4-3-8",
            "screenshot_dialog_load_video.png": "Fig 4-3-9",
            "screenshot_step_stop_camera.png": "Fig 4-3-10",
        }
        merged: dict[str, dict] = {}
        for r in rows:
            merged[r["file"]] = r
        print("\n" + "=" * 100)
        print(f"{'Ref':<12} {'File':<36} {'Status':<16} Note")
        print("=" * 100)
        out_abs = os.path.abspath(args.output_dir)
        for fn, _ in expected:
            r = merged.get(fn, {})
            disk = os.path.isfile(os.path.join(out_abs, fn))
            st = r.get("status", "ok" if disk else "missing")
            note = r.get("note", "")
            extra = ""
            if fn == "screenshot_camera_permission.png" and disk:
                if "placeholder" in note or st.startswith("ok"):
                    extra = " (replace with OS permission UI if required)"
            print(f"{legend[fn]:<12} {fn:<36} {st:<16} {note}{extra}")
        print("=" * 100)

        pngs = _list_png(args.output_dir)
        rel = os.path.relpath(args.output_dir, ROOT)
        print(f"\n{rel}/*.png: {len(pngs)}")
        for p in pngs:
            print(" ", p)

        print("\nDone.")
        return 0

    print("Usage:  python scripts/capture_manual_screenshots.py --auto --native")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
