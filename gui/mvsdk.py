"""
Python ctypes wrapper for MindVision / HuaTengVision Camera SDK.

Platform support
----------------
macOS  (arm64)    : libmvsdk.dylib  –  copied into gui/ and ad-hoc signed
Windows (x86-64)  : MVCAMSDK_X64.dll + companion DLLs – bundled in gui/win_sdk/

Key API flow
------------
    sdk_init()
    devs = enumerate_devices()
    handle = camera_init(devs[0])
    cap = get_capability(handle)
    set_isp_out_format(handle, CAMERA_MEDIA_TYPE_MONO8 or RGB8)
    buf = align_malloc(max_width * max_height * 3, 16)
    set_trigger_mode(handle, 0)   # continuous
    camera_play(handle)
    loop:
        head, raw = get_image_buffer(handle, timeout_ms=200)
        image_process(handle, raw, buf, head)
        release_image_buffer(handle, raw)
        frame = frame_to_numpy(buf, head)
    camera_stop(handle)
    camera_uninit(handle)
    align_free(buf)

Windows notes
-------------
All required DLLs are pre-bundled in gui/win_sdk/:
    MVCAMSDK_X64.dll, MVImageProcess_X64.DLL,
    hAcqHuaTengVision*_X64.dll, Usb2/Usb3*.Interface

The USB driver (MvU2Camera.inf/.sys) must be installed once per machine.
Driver files are in gui/win_sdk/drivers/.  Install via Device Manager →
right-click the camera → Update Driver → Browse → point to that folder.
"""

import ctypes
import os
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

CAMERA_STATUS_SUCCESS       = 0

CAMERA_MEDIA_TYPE_MONO8     = 0x01080001  # mono 8-bit
CAMERA_MEDIA_TYPE_RGB8      = 0x02180014  # color 24-bit RGB

MAX_DEVICES = 16

# ── Struct definitions ─────────────────────────────────────────────────────────

class tSdkCameraDevInfo(ctypes.Structure):
    """Device enumeration info returned by CameraEnumerateDevice."""
    _fields_ = [
        ("acProductSeries", ctypes.c_char * 32),
        ("acProductName",   ctypes.c_char * 32),
        ("acFriendlyName",  ctypes.c_char * 32),
        ("acLinkName",      ctypes.c_char * 32),
        ("acDriverVersion", ctypes.c_char * 32),
        ("acSensorType",    ctypes.c_char * 32),
        ("acPortType",      ctypes.c_char * 32),
        ("acSn",            ctypes.c_char * 32),
        ("uInstance",       ctypes.c_uint),
    ]

    @property
    def friendly_name(self) -> str:
        return self.acFriendlyName.decode("utf-8", errors="replace")

    @property
    def product_name(self) -> str:
        return self.acProductName.decode("utf-8", errors="replace")

    @property
    def sn(self) -> str:
        return self.acSn.decode("utf-8", errors="replace")


class tSdkFrameHead(ctypes.Structure):
    """Frame header returned with every captured image."""
    _fields_ = [
        ("uiMediaType",   ctypes.c_uint),
        ("uBytes",        ctypes.c_uint),
        ("iWidth",        ctypes.c_int),
        ("iHeight",       ctypes.c_int),
        ("iWidthZoomSw",  ctypes.c_int),
        ("iHeightZoomSw", ctypes.c_int),
        ("bIsTrigger",    ctypes.c_int),
        ("uiTimeStamp",   ctypes.c_uint),
        ("uiExpTime",     ctypes.c_uint),
        ("fAnalogGain",   ctypes.c_float),
        ("iGamma",        ctypes.c_int),
        ("iContrast",     ctypes.c_int),
        ("iSaturation",   ctypes.c_int),
        ("fRgain",        ctypes.c_float),
        ("fGgain",        ctypes.c_float),
        ("fBgain",        ctypes.c_float),
    ]


# ── Embedded structs for tSdkCameraCapbility ──────────────────────────────────

class _tSdkExpose(ctypes.Structure):
    _fields_ = [
        ("uiTargetMin",     ctypes.c_uint),
        ("uiTargetMax",     ctypes.c_uint),
        ("uiAnalogGainMin", ctypes.c_uint),
        ("uiAnalogGainMax", ctypes.c_uint),
        ("fAnalogGainStep", ctypes.c_float),
        ("uiExposeTimeMin", ctypes.c_uint),
        ("uiExposeTimeMax", ctypes.c_uint),
    ]


class _tSdkResolutionRange(ctypes.Structure):
    _fields_ = [
        ("iHeightMax",          ctypes.c_int),
        ("iHeightMin",          ctypes.c_int),
        ("iWidthMax",           ctypes.c_int),
        ("iWidthMin",           ctypes.c_int),
        ("uSkipModeMask",       ctypes.c_uint),
        ("uBinSumModeMask",     ctypes.c_uint),
        ("uBinAverageModeMask", ctypes.c_uint),
        ("uResampleMask",       ctypes.c_uint),
    ]


class _tRgbGainRange(ctypes.Structure):
    _fields_ = [
        ("iRGainMin", ctypes.c_int), ("iRGainMax", ctypes.c_int),
        ("iGGainMin", ctypes.c_int), ("iGGainMax", ctypes.c_int),
        ("iBGainMin", ctypes.c_int), ("iBGainMax", ctypes.c_int),
    ]


class _tRange2(ctypes.Structure):
    _fields_ = [("iMin", ctypes.c_int), ("iMax", ctypes.c_int)]


class _tSdkIspCapacity(ctypes.Structure):
    _fields_ = [
        ("bMonoSensor",        ctypes.c_int),
        ("bWbOnce",            ctypes.c_int),
        ("bAutoWb",            ctypes.c_int),
        ("bAutoExposure",      ctypes.c_int),
        ("bManualExposure",    ctypes.c_int),
        ("bAntiFlick",         ctypes.c_int),
        ("bDeviceIsp",         ctypes.c_int),
        ("bForceUseDeviceIsp", ctypes.c_int),
        ("bZoomHD",            ctypes.c_int),
    ]


class tSdkCameraCapbility(ctypes.Structure):
    """
    Camera capability descriptor.
    ctypes automatically inserts alignment padding between pointer and int
    fields according to the arm64 ABI, matching the C struct layout exactly.
    """
    _fields_ = [
        ("pTriggerDesc",       ctypes.c_void_p),
        ("iTriggerDesc",       ctypes.c_int),
        ("pImageSizeDesc",     ctypes.c_void_p),
        ("iImageSizeDesc",     ctypes.c_int),
        ("pClrTempDesc",       ctypes.c_void_p),
        ("iClrTempDesc",       ctypes.c_int),
        ("pMediaTypeDesc",     ctypes.c_void_p),
        ("iMediaTypdeDesc",    ctypes.c_int),
        ("pFrameSpeedDesc",    ctypes.c_void_p),
        ("iFrameSpeedDesc",    ctypes.c_int),
        ("pPackLenDesc",       ctypes.c_void_p),
        ("iPackLenDesc",       ctypes.c_int),
        ("iOutputIoCounts",    ctypes.c_int),
        ("iInputIoCounts",     ctypes.c_int),
        ("pPresetLutDesc",     ctypes.c_void_p),
        ("iPresetLut",         ctypes.c_int),
        ("iUserDataMaxLen",    ctypes.c_int),
        ("bParamInDevice",     ctypes.c_int),
        ("pAeAlmSwDesc",       ctypes.c_void_p),
        ("iAeAlmSwDesc",       ctypes.c_int),
        ("pAeAlmHdDesc",       ctypes.c_void_p),
        ("iAeAlmHdDesc",       ctypes.c_int),
        ("pBayerDecAlmSwDesc", ctypes.c_void_p),
        ("iBayerDecAlmSwDesc", ctypes.c_int),
        ("pBayerDecAlmHdDesc", ctypes.c_void_p),
        ("iBayerDecAlmHdDesc", ctypes.c_int),
        ("sExposeDesc",        _tSdkExpose),
        ("sResolutionRange",   _tSdkResolutionRange),
        ("sRgbGainRange",      _tRgbGainRange),
        ("sSaturationRange",   _tRange2),
        ("sGammaRange",        _tRange2),
        ("sContrastRange",     _tRange2),
        ("sSharpnessRange",    _tRange2),
        ("sIspCapacity",       _tSdkIspCapacity),
    ]


# ── Library loading ────────────────────────────────────────────────────────────

import sys as _sys
import platform as _platform

_HERE = os.path.dirname(__file__)

def _sdk_search_paths() -> list[str]:
    """
    Return platform-specific candidate paths for the MindVision SDK library.

    macOS   → libmvsdk.dylib  (arm64, provided in gui/)
    Windows → MVCAMSDK_X64.dll (64-bit) or MVCAMSDK.dll (32-bit)
    Linux   → libMVSDK.so (not officially supported)
    """
    if _sys.platform == "darwin":
        return [
            os.path.join(_HERE, "libmvsdk.dylib"),
            "/usr/local/lib/libmvsdk.dylib",
            os.path.expanduser("~/Downloads/Mac_sdk_m3(250120)/lib/libmvsdk.dylib"),
        ]
    elif _sys.platform == "win32":
        arch = _platform.architecture()[0]   # '64bit' or '32bit'
        dll_name = "MVCAMSDK_X64.dll" if arch == "64bit" else "MVCAMSDK.dll"
        win_sdk  = os.path.join(_HERE, "win_sdk")   # bundled DLLs shipped with project
        return [
            # 1. Bundled win_sdk/ subfolder (shipped with the project – preferred)
            os.path.join(win_sdk, dll_name),
            # 2. Placed directly next to mvsdk.py (legacy / manual copy)
            os.path.join(_HERE, dll_name),
            # 3. Standard MindVision Windows installer locations
            rf"C:\Program Files\MindVision\SDK\{dll_name}",
            rf"C:\Program Files (x86)\MindVision\SDK\{dll_name}",
            rf"C:\MindVision\SDK\{dll_name}",
            # 4. System32 (if user ran the SDK installer's system-wide option)
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", dll_name),
        ]
    else:
        return [
            os.path.join(_HERE, "libMVSDK.so"),
            "/usr/lib/libMVSDK.so",
            "/usr/local/lib/libMVSDK.so",
        ]


_lib: ctypes.CDLL | None = None


def _find_lib() -> str:
    paths = _sdk_search_paths()
    for p in paths:
        if os.path.isfile(p):
            return p
    platform_name = {"darwin": "macOS", "win32": "Windows"}.get(_sys.platform, _sys.platform)
    raise FileNotFoundError(
        f"MindVision SDK library not found on {platform_name}.\n"
        "Expected locations:\n" + "\n".join(f"  {p}" for p in paths) + "\n\n"
        "macOS  : copy libmvsdk.dylib into gui/, then run:\n"
        "           codesign --force --sign - gui/libmvsdk.dylib\n"
        "Windows: DLLs should be bundled in gui/win_sdk/ (MVCAMSDK_X64.dll etc.).\n"
        "         If missing, re-extract them from the project archive or\n"
        "         install the HuaTengVision SDK and copy the DLLs manually."
    )


def _setup(lib: ctypes.CDLL) -> None:
    """Declare argtypes / restype for every function we use."""
    c_int      = ctypes.c_int
    c_uint     = ctypes.c_uint
    c_void_p   = ctypes.c_void_p
    c_double   = ctypes.c_double
    c_double_p = ctypes.POINTER(ctypes.c_double)

    lib.CameraSdkInit.restype  = c_int
    lib.CameraSdkInit.argtypes = [c_int]

    lib.CameraEnumerateDevice.restype  = c_int
    lib.CameraEnumerateDevice.argtypes = [
        ctypes.POINTER(tSdkCameraDevInfo),
        ctypes.POINTER(c_int),
    ]

    lib.CameraInit.restype  = c_int
    lib.CameraInit.argtypes = [
        ctypes.POINTER(tSdkCameraDevInfo),
        c_int, c_int,
        ctypes.POINTER(c_int),
    ]

    lib.CameraGetCapability.restype  = c_int
    lib.CameraGetCapability.argtypes = [
        c_int,
        ctypes.POINTER(tSdkCameraCapbility),
    ]

    lib.CameraSetIspOutFormat.restype  = c_int
    lib.CameraSetIspOutFormat.argtypes = [c_int, c_uint]

    lib.CameraAlignMalloc.restype  = c_void_p
    lib.CameraAlignMalloc.argtypes = [c_int, c_int]

    lib.CameraAlignFree.restype  = None
    lib.CameraAlignFree.argtypes = [c_void_p]

    lib.CameraSetAeState.restype  = c_int
    lib.CameraSetAeState.argtypes = [c_int, c_int]

    lib.CameraSetExposureTime.restype  = c_int
    lib.CameraSetExposureTime.argtypes = [c_int, c_double]

    lib.CameraGetExposureTime.restype  = c_int
    lib.CameraGetExposureTime.argtypes = [c_int, c_double_p]

    lib.CameraSetAnalogGain.restype  = c_int
    lib.CameraSetAnalogGain.argtypes = [c_int, c_int]

    lib.CameraGetAnalogGain.restype  = c_int
    lib.CameraGetAnalogGain.argtypes = [c_int, ctypes.POINTER(c_int)]

    lib.CameraSetTriggerMode.restype  = c_int
    lib.CameraSetTriggerMode.argtypes = [c_int, c_int]

    lib.CameraPlay.restype  = c_int
    lib.CameraPlay.argtypes = [c_int]

    lib.CameraStop.restype  = c_int
    lib.CameraStop.argtypes = [c_int]

    lib.CameraGetImageBuffer.restype  = c_int
    lib.CameraGetImageBuffer.argtypes = [
        c_int,
        ctypes.POINTER(tSdkFrameHead),
        ctypes.POINTER(c_void_p),
        c_uint,
    ]

    lib.CameraImageProcess.restype  = c_int
    lib.CameraImageProcess.argtypes = [c_int, c_void_p, c_void_p, ctypes.POINTER(tSdkFrameHead)]

    lib.CameraReleaseImageBuffer.restype  = c_int
    lib.CameraReleaseImageBuffer.argtypes = [c_int, c_void_p]

    lib.CameraUnInit.restype  = c_int
    lib.CameraUnInit.argtypes = [c_int]


def _ensure_dll_dirs() -> None:
    """
    Windows only: register extra DLL search directories so ctypes can find
    MVCAMSDK_X64.dll and its companion DLLs (MVImageProcess, hAcqHuaTengVision*,
    Usb*Camera*.Interface).

    Python 3.8+ no longer searches PATH for DLLs by default, so we use
    os.add_dll_directory().  Falls back to PATH injection on older Python.

    We always register gui/win_sdk/ (the bundled folder) first, then check
    every other candidate directory in the search path list.
    """
    if _sys.platform != "win32":
        return

    def _add(folder: str) -> None:
        if not os.path.isdir(folder):
            return
        try:
            os.add_dll_directory(folder)            # Python 3.8+
        except AttributeError:
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")

    # Always add the bundled win_sdk/ first so companion DLLs are found.
    _add(os.path.join(_HERE, "win_sdk"))

    # Then add every candidate directory from the search-path list.
    for p in _sdk_search_paths():
        _add(os.path.dirname(p))


def get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _ensure_dll_dirs()
        path = _find_lib()
        # MindVision SDK uses __cdecl on both macOS and Windows x64.
        _lib = ctypes.CDLL(path)
        _setup(_lib)
    return _lib


# ── High-level helpers ─────────────────────────────────────────────────────────

def sdk_init(lang: int = 0) -> None:
    """Initialize the SDK (call once per process, lang=0 English, 1 Chinese)."""
    lib = get_lib()
    lib.CameraSdkInit(lang)


def enumerate_devices() -> list[tSdkCameraDevInfo]:
    """Return a list of connected MindVision cameras."""
    lib = get_lib()
    arr = (tSdkCameraDevInfo * MAX_DEVICES)()
    count = ctypes.c_int(MAX_DEVICES)
    lib.CameraEnumerateDevice(arr, ctypes.byref(count))
    return list(arr[: count.value])


def camera_init(dev_info: tSdkCameraDevInfo) -> int:
    """Open camera; returns handle (> 0) on success."""
    lib = get_lib()
    handle = ctypes.c_int(0)
    status = lib.CameraInit(ctypes.byref(dev_info), -1, -1, ctypes.byref(handle))
    if status != CAMERA_STATUS_SUCCESS:
        raise RuntimeError(f"CameraInit failed, status={status}")
    return handle.value


def get_capability(handle: int) -> tSdkCameraCapbility:
    lib = get_lib()
    cap = tSdkCameraCapbility()
    lib.CameraGetCapability(handle, ctypes.byref(cap))
    return cap


def set_isp_out_format(handle: int, fmt: int) -> None:
    get_lib().CameraSetIspOutFormat(handle, fmt)


def align_malloc(size: int, align: int = 16) -> int:
    """Allocate aligned memory; returns address (int)."""
    addr = get_lib().CameraAlignMalloc(size, align)
    if not addr:
        raise MemoryError("CameraAlignMalloc failed")
    return addr


def align_free(addr: int) -> None:
    if addr:
        get_lib().CameraAlignFree(addr)


def set_ae_state(handle: int, enable: bool) -> None:
    get_lib().CameraSetAeState(handle, int(enable))


def set_exposure_time(handle: int, us: float) -> None:
    """Set exposure time in microseconds."""
    get_lib().CameraSetExposureTime(handle, ctypes.c_double(us))


def get_exposure_time(handle: int) -> float:
    """Return current exposure time in microseconds."""
    v = ctypes.c_double(0.0)
    get_lib().CameraGetExposureTime(handle, ctypes.byref(v))
    return v.value


def set_analog_gain(handle: int, gain: int) -> None:
    get_lib().CameraSetAnalogGain(handle, gain)


def get_analog_gain(handle: int) -> int:
    v = ctypes.c_int(0)
    get_lib().CameraGetAnalogGain(handle, ctypes.byref(v))
    return v.value


def set_trigger_mode(handle: int, mode: int) -> None:
    """0 = continuous, 1 = software trigger."""
    get_lib().CameraSetTriggerMode(handle, mode)


def camera_play(handle: int) -> None:
    get_lib().CameraPlay(handle)


def camera_stop(handle: int) -> None:
    get_lib().CameraStop(handle)


def camera_uninit(handle: int) -> None:
    get_lib().CameraUnInit(handle)


def get_image_buffer(handle: int, timeout_ms: int = 200):
    """
    Grab one raw frame.

    Returns (tSdkFrameHead, raw_ptr_int) on success, or (None, None) on timeout.
    Caller MUST call release_image_buffer(handle, raw_ptr_int) after processing.
    """
    lib = get_lib()
    head    = tSdkFrameHead()
    raw_ptr = ctypes.c_void_p(0)
    status  = lib.CameraGetImageBuffer(
        handle,
        ctypes.byref(head),
        ctypes.byref(raw_ptr),
        ctypes.c_uint(timeout_ms),
    )
    if status != CAMERA_STATUS_SUCCESS:
        return None, None
    return head, raw_ptr.value


def image_process(handle: int, raw_ptr: int, out_buf: int, head: tSdkFrameHead) -> bool:
    """Run ISP (demosaic / format convert) from raw_ptr → out_buf. Returns True on success."""
    status = get_lib().CameraImageProcess(
        handle,
        ctypes.c_void_p(raw_ptr),
        ctypes.c_void_p(out_buf),
        ctypes.byref(head),
    )
    return status == CAMERA_STATUS_SUCCESS


def release_image_buffer(handle: int, raw_ptr: int) -> None:
    get_lib().CameraReleaseImageBuffer(handle, ctypes.c_void_p(raw_ptr))


def frame_to_numpy(out_buf: int, head: tSdkFrameHead) -> np.ndarray:
    """
    Convert the ISP-processed buffer to a numpy array.

    Returns:
        MONO8 → (H, W)     uint8 grayscale
        RGB8  → (H, W, 3)  uint8 BGR  (flipped from SDK's RGB for OpenCV/Qt compat)
    """
    w, h   = head.iWidth, head.iHeight
    nbytes = head.uBytes
    arr = np.frombuffer(
        (ctypes.c_ubyte * nbytes).from_address(out_buf),
        dtype=np.uint8,
    ).copy()

    if head.uiMediaType == CAMERA_MEDIA_TYPE_MONO8:
        return arr.reshape(h, w)
    else:
        rgb = arr.reshape(h, w, 3)
        return rgb[:, :, ::-1]  # RGB → BGR
