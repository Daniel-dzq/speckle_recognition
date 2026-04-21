# GUI Tutorial --- Speckle-PUF Live Demo

This document explains how to install, launch, and operate the Speckle-PUF live
demo GUI. The application is a single-window control centre for real-time optical
experiments.

It lets you:

- show letters or patterns on an SLM display,
- capture live speckle output from a CCD camera (MindVision HT-UBS300C or any webcam),
- run real-time deep learning recognition,
- present everything on a multi-monitor setup.

---

## Table of Contents

1. [Requirements and installation](#1-requirements-and-installation)
2. [Hardware overview](#2-hardware-overview)
3. [Launch the GUI](#3-launch-the-gui)
4. [Window layout](#4-window-layout)
5. [Fiber & Model section](#5-fiber--model-section)
6. [SLM Output Window section](#6-slm-output-window-section)
7. [Camera / Video Source section](#7-camera--video-source-section)
8. [Camera Settings panel](#8-camera-settings-panel)
9. [Inference Settings section](#9-inference-settings-section)
10. [Recognition Output panel](#10-recognition-output-panel)
11. [Status bar](#11-status-bar)
12. [Complete demo workflow](#12-complete-demo-workflow)
13. [Troubleshooting](#13-troubleshooting)
14. [Best practices](#14-best-practices)

---

## 1. Requirements and installation

### Python environment

```bash
conda activate recognition        # or your own environment name
pip install -r requirements.txt   # PySide6, opencv-python, torch, etc.
```

### Environment check

Run this before the first use to verify all dependencies:

```bash
python scripts/env_check.py
```

Expected output (example on macOS with MindVision CCD ready):

```
============================================================
  Speckle-PUF Environment Check
============================================================
            Python : 3.11.x
    MindVision SDK : OK  ---  1 camera(s) detected
            OpenCV : 4.x.x
           PyTorch : 2.x.x
           PySide6 : 6.x.x
============================================================
```

### MindVision CCD SDK setup

The SDK libraries are already bundled in the repository.

| Platform | Library location | Status |
|----------|-----------------|--------|
| macOS (Apple Silicon M1--M4) | `gui/libmvsdk.dylib` | **Included --- ready to use** |
| Windows 64-bit | `gui/win_sdk/MVCAMSDK_X64.dll` + companion DLLs | **Included --- ready to use** |

**macOS** --- no action needed. The library is already signed and ready.

**Windows** --- one-time USB driver installation:

1. Plug in the HT-UBS300C camera via USB.
2. Open **Device Manager**. The camera may appear as "Unknown Device".
3. Right-click the camera -> **Update Driver** -> **Browse my computer for drivers**.
4. Point to the folder: `gui/win_sdk/drivers/`.
5. After installation, the camera should appear as a MindVision device.

All SDK DLLs (`MVCAMSDK_X64.dll`, `MVImageProcess_X64.DLL`, `hAcqHuaTengVision*_X64.dll`,
`Usb*Camera*.Interface`) are pre-bundled in `gui/win_sdk/` and loaded automatically.
No manual DLL copying is needed.

> **Note**: On Windows, after driver installation the HT-UBS300C also registers
> as a DirectShow device. This means the standard **Start Camera** button
> (OpenCV path) often works directly without using the MindVision SDK button.

---

## 2. Hardware overview

The typical setup consists of:

| Component | Role |
|-----------|------|
| Computer (macOS or Windows) | Runs the GUI |
| SLM monitor (connected via HDMI/DP) | Displays letters or phase patterns |
| MindVision HT-UBS300C (USB) | Captures speckle images |
| Multimode / plastic optical fibre | Transmits light between SLM and CCD |

The GUI handles all of these from a single window.

---

## 3. Launch the GUI

From the project root directory:

```bash
python scripts/launch_demo.py
```

At startup the terminal prints a banner like this:

```
================================================================
  Speckle-PUF Demo  |  starting up
================================================================
  Python     : 3.11.x
  Platform   : macOS-14.x (arm64)
  OpenCV     : 4.x.x
  MindVision : [OK] OK
================================================================
```

If MindVision shows `[--] NOT FOUND`, see [Section 1](#1-requirements-and-installation)
for setup instructions. The GUI still opens normally even without the MindVision
SDK --- you can use a webcam or video file instead.

---

## 4. Window layout

```
+------------------------------+-------------------------------+
|  LEFT PANEL (controls)       |  RIGHT PANEL (live output)    |
|                              |                               |
|  - Fiber & Model             |  - Live Camera Feed           |
|  - SLM Output Window         |                               |
|  - Camera / Video Source     |  - Recognition Output         |
|  - Camera Settings           |    - Instant prediction       |
|  - Inference Settings        |    - Confidence               |
|                              |    - Smoothed (majority vote) |
|                              |    - Top-5 candidates         |
+------------------------------+-------------------------------+
|  LOG BOX  |  STATUS BAR  (Device - Model - FPS)              |
+--------------------------------------------------------------+
```

---

## 5. Fiber & Model section

This section selects which trained recognition model to use.

### Controls

| Control | Description |
|---------|-------------|
| **Fiber** dropdown | Lists all fibers found in `video_capture/`. Select the one matching your optical setup. |
| **Refresh** | Rescans `video_capture/` for new fibers. |
| **Checkpoint** dropdown | Lists `.pth` files in `checkpoints/`. The GUI tries to pre-select the file matching the chosen fiber. |
| **Load Model** | Loads the selected checkpoint into the inference engine. |

After a successful load the model status changes to **Loaded** and the status bar
is updated.

### Typical workflow

1. Select the correct fiber.
2. Select the matching checkpoint.
3. Click **Load Model**.
4. Confirm the log shows `Model loaded successfully`.

### Troubleshooting

- **No checkpoints found** --- place your trained `.pth` files in `checkpoints/`
  and name them ending in `_best.pth`.
- **Model load failed** --- check the log for the exact error. Common causes:
  wrong fiber/checkpoint pair, environment mismatch, or corrupted file.

---

## 6. SLM Output Window section

This section controls the SLM display in a multi-monitor setup.

### Controls

| Control | Description |
|---------|-------------|
| **Open SLM Window** | Opens (or hides) the SLM output window. |
| **SLM screen** dropdown | Lists all monitors detected by Qt with name, resolution, and position. Select the SLM monitor here. |
| **Refresh** | Rescans connected monitors. Use this after plugging in the SLM cable. |
| **Fullscreen on selected screen** | When checked, the SLM window is placed fullscreen on the chosen monitor. |
| **Move SLM to Selected Screen** | Forces the SLM window onto the selected monitor immediately. |
| **Letter** field | Type the letter (A--Z) to display. |
| **Send to SLM** | Displays the typed letter on the SLM screen. |
| **Font size** | Adjusts the letter size. Increase if the projected pattern is too small. |
| **Prev / Next** | Cycles through A--Z without typing. Convenient for fast demos. |
| **Load Image to SLM** | Opens a file dialog and displays an image (PNG/JPG/BMP/TIFF) on the SLM. |

### Workflow for letter display

1. Connect the SLM monitor.
2. Click **Refresh** in this section.
3. Select the SLM monitor in the **SLM screen** dropdown.
4. Enable **Fullscreen on selected screen**.
5. Click **Move SLM to Selected Screen**.
6. Type a letter and click **Send to SLM**.

### Workflow for image / phase pattern display

1. Select the SLM monitor and move the window to it.
2. Click **Load Image to SLM**.
3. Choose your PNG/BMP/TIFF pattern file.

---

## 7. Camera / Video Source section

This section controls the live frame input.

### Option A --- MindVision CCD (HT-UBS300C) via vendor SDK

> This is the recommended option when using the HT-UBS300C CCD on macOS.
> It bypasses OpenCV and communicates with the camera through the MindVision SDK.

**Steps:**

1. Plug the HT-UBS300C into a USB port and wait about 3 seconds.
2. Click **MindVision CCD (HT-UBS300C)**.
3. The GUI enumerates MindVision devices automatically.
4. If a camera is found, the live speckle feed starts immediately.

The status bar shows `MindVision CCD: <camera name>` and the log prints the
camera's friendly name and serial number.

**If no camera is detected**, a dialog appears with a platform-specific checklist:

- macOS: check USB connection and System Settings -> Privacy -> Camera.
- Windows: verify the camera appears in Device Manager; if shown as "Unknown
  Device", install the USB driver from `gui/win_sdk/drivers/` first.

> **Windows shortcut**: On Windows, after driver installation the HT-UBS300C
> is accessible via OpenCV's DirectShow backend. You can skip the MindVision
> button and use **Start Camera** directly. Both paths work.

### Option B --- Standard webcam or capture card via OpenCV

Use this for regular webcams, USB capture cards, or any OpenCV-compatible device.

| Control | Description |
|---------|-------------|
| **Camera index** | Device index used by OpenCV. `0` is usually the default camera. |
| **Scan Available Cameras** | Probes indices 0--9 and lists working devices. On macOS this also triggers the system camera-permission dialog. |
| **Resolution** dropdown | Requests a specific resolution from OpenCV. |
| **Start Camera** | Opens the camera and starts the live feed. |
| **Stop** | Stops the active capture (works for all source types). |

### Option C --- Local video file

Use this for offline testing or when no camera is connected.

| Control | Description |
|---------|-------------|
| **Load Video File** | Opens a file dialog to select AVI/MP4/MKV/MOV. The video loops automatically. |

### Source label

Displays the currently active input source, for example:

- `MindVision CCD: HT-UBS300C`
- `Camera device: 0`
- `File: A_demo.avi`

---

## 8. Camera Settings panel

This panel adjusts parameters of the live camera in real time. Settings apply to
whichever camera backend is active (MindVision or OpenCV).

| Control | Description |
|---------|-------------|
| **Auto Exposure** | Enables automatic exposure control. When unchecked, the Exposure slider is active. |
| **Exposure** | Manual exposure value. For MindVision this is in microseconds. For OpenCV it is a log2 scale. |
| **Gain** | Analogue gain. |
| **Brightness / Contrast / Saturation / Gamma / Sharpness** | Standard image parameters (availability depends on the camera model). |
| **Auto White Balance** | Enables automatic white balance. |
| **White Balance Temp** | Colour temperature in Kelvin (for colour cameras). |
| **Flip H / Flip V** | Mirror the frame horizontally or vertically in software. |

> For the HT-UBS300C monochrome CCD, white balance and saturation controls have
> no effect.

---

## 9. Inference Settings section

| Control | Description |
|---------|-------------|
| **Infer every N frames** | Runs inference once every N captured frames. A value of `4` is a good default. |
| **Vote window** | Number of recent predictions used for majority-vote smoothing. |
| **Recognition active** | Checkbox to enable or pause inference while keeping the camera feed running. |

---

## 10. Recognition Output panel

| Field | Description |
|-------|-------------|
| **Instant** | Top-1 prediction from the latest inference call. |
| **Confidence** | Softmax probability of the top-1 class. |
| **Smoothed (majority vote)** | Most frequent prediction over the last *Vote window* inferences. Use this for presentation. |
| **Top-5 candidates** | Top-5 classes and their probabilities. |

---

## 11. Status bar

| Field | Description |
|-------|-------------|
| **Device** | Compute device used for inference: `CUDA`, `MPS` (Apple Silicon GPU), or `CPU`. |
| **Model** | Name of the loaded checkpoint. |
| **FPS** | Current camera frame rate. |

> On Apple Silicon (M1--M4) the inference device will show `MPS`, which means
> the onboard GPU is being used for acceleration.

---

## 12. Complete demo workflow

Follow this order for a full optical experiment:

```
 1.  Connect the HT-UBS300C via USB.
 2.  Connect the SLM monitor via HDMI or DisplayPort.
 3.  Run:  python scripts/launch_demo.py
 4.  Confirm the terminal banner shows  MindVision : [OK].
 5.  Select the correct fiber in the Fiber & Model section.
 6.  Load the matching checkpoint.
 7.  Click Refresh in the SLM section, then select the SLM monitor.
 8.  Enable Fullscreen, then click Move SLM to Selected Screen.
 9.  Click MindVision CCD (HT-UBS300C) to start the live feed.
     (On Windows you can also use Start Camera directly.)
10.  Confirm speckle frames appear in the Live Camera Feed panel.
11.  Send a letter to the SLM (or use Prev / Next to cycle).
12.  Enable Recognition active.
13.  Read the Smoothed prediction --- that is the real-time recognition result.
```

---

## 13. Troubleshooting

### MindVision camera not detected

| Platform | Check |
|----------|-------|
| macOS | USB connection solid? Wait 3 s then click the button again. |
| macOS | System Settings -> Privacy & Security -> Camera -> grant access to Terminal/Python. |
| Windows | Device Manager -> camera listed? If "Unknown Device", install the USB driver from `gui/win_sdk/drivers/`. |
| Windows | `gui/win_sdk/MVCAMSDK_X64.dll` present? If not, re-extract from the project archive. |

### Camera feed is black or frozen

- Wrong exposure --- reduce exposure or enable Auto Exposure.
- Cable or connector issue --- replug the USB and click the button again.
- For OpenCV cameras: wrong device index --- click Scan Available Cameras.

### SLM monitor shows nothing

1. Click **Refresh** in the SLM section.
2. Re-select the correct monitor.
3. Click **Move SLM to Selected Screen**.
4. Click **Send to SLM** again.

### SLM window opened on the wrong monitor

Use **Move SLM to Selected Screen** inside the GUI. Do not drag it manually.

### Image on SLM is not scaled correctly

Reload the image after the window is already positioned on the correct screen.

### Recognition does not start

Check all three conditions:

- Model status shows **Loaded**.
- Camera feed is running (frames visible in the live panel).
- **Recognition active** checkbox is ticked.

### Prediction is unstable or confidence is low

- Increase **Vote window**.
- Verify you loaded the checkpoint that matches the active fiber.
- Check optical alignment and reduce vibration.
- Adjust camera exposure so the speckle pattern is clearly visible (not over- or
  under-exposed).

### macOS: `libmvsdk.dylib` load error --- "code signature not valid"

Run once in Terminal:

```bash
codesign --force --sign - gui/libmvsdk.dylib
```

Then relaunch the GUI.

### Windows: DLL load error

- Ensure `gui/win_sdk/` contains all required DLLs (see README for the full list).
- Ensure you are using 64-bit Python on a 64-bit Windows installation.
- Try running `python scripts/env_check.py` to diagnose.

---

## 14. Best practices

1. **Always use the MindVision CCD button for the HT-UBS300C on macOS.**
   The "Start Camera" button uses OpenCV, which cannot access this camera on macOS.
   On Windows, both buttons work.

2. **Always select the SLM monitor inside the GUI.**
   Do not rely on OS display duplication settings alone.

3. **Load the correct checkpoint for the active fiber.**
   A cross-fiber mismatch will produce poor recognition even if everything else
   is working.

4. **Use Smoothed prediction for demonstrations.**
   It is more stable than the raw instant top-1 output.

5. **Check the log box whenever something goes wrong.**
   Most errors are reported there with a specific message.

6. **Use a video file when no camera is connected.**
   `Load Video File` lets you test the full recognition pipeline offline.

7. **Run `env_check.py` first on a new machine.**
   It shows exactly which dependencies and SDK files are present before you start
   the experiment.

8. **Share pre-trained models to skip training.**
   Colleagues only need the `gui/` folder, `scripts/launch_demo.py`,
   `checkpoints/*.pth`, `models.py`, `dataset.py`, and `requirements.txt` to run
   the demo. Training data (`video_capture/`) is not needed.
