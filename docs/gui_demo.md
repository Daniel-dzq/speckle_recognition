# GUI demo

## Run

From the release root:

```bash
pip install -r requirements.txt
python scripts/launch_demo.py
```

## Layout

- **Left** — Challenge input preview (PPT-exported image or text).
- **Center** — Live speckle camera view.
- **Right** — Recognition result and robot authorization panel.

## Challenge set

On startup, `challenge_inputs/manifest.json` loads automatically. Use **Prev / Next** to cycle challenges, **Send to SLM** to display the pattern.

Re-export from PowerPoint:

```bash
python scripts/export_ppt_challenges.py --input input.pptx --output_dir challenge_inputs
```

## Models

Select **Fiber1–Fiber15** under “Authorized model”. Checkpoints:

`models/final_15fibers/FiberN.pth`

Label map:

`models/final_15fibers/label_map.json`

## Label matching

Comparison is case-insensitive for letters (`A` matches `a`). Digits and `boy` / `girl` are compared as normalized strings.

## Camera backends

### macOS

- Bundled `gui/libmvsdk.dylib` for MindVision HT-series USB cameras.
- If the library fails to load, use a webcam via OpenCV (GUI still opens).
- You may need to codesign the dylib:  
  `codesign --force --sign - gui/libmvsdk.dylib`

### Windows

- Bundled SDK DLLs under `gui/win_sdk/`.
- Install the MindVision **camera driver** from the vendor if the device is not detected.
- `launch_demo.py` adds `gui/win_sdk/` to the DLL search path on Windows.

### All platforms

- OpenCV webcam / video file modes remain available when MindVision is not used.

## SLM

Use **Open SLM Window**, select the display, then **Send to SLM** for the current challenge image.
