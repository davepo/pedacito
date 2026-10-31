"""Optional screenshot capture and textual analysis.

Usage
-----
    from . import vision

    ok, how = vision.capture(out_png)
    report = vision.analyze(out_png, previous=old_png)
    desc = vision.describe(client, cfg, out_png)

A text-only coding model cannot see a GUI. This module turns a screenshot into
text it can act on:

  * pixel statistics (mean brightness, near-black/white share, unique colors),
  * a blank/black-screen verdict,
  * OCR text when `tesseract` is installed,
  * an optional description from a multimodal model (`vision_model`).

Pillow is an optional dependency (`pip install pedacito-local[vision]`). It is
needed for pixel analysis and for the portable ImageGrab capture path; without
it, capture can still use native tools and OCR still works.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout)
    except Exception as e:                       # pragma: no cover - env specific
        return False, str(e)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    return p.returncode == 0, out


def _pillow():
    try:
        from PIL import Image, ImageGrab
        return Image, ImageGrab
    except Exception:
        return None


def capture(path) -> tuple[bool, str]:
    """Capture the current screen to `path` (PNG). Returns (ok, method/error)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pillow = _pillow()
    if pillow is not None:
        _, ImageGrab = pillow
        try:
            ImageGrab.grab().save(path, "PNG")
            return True, "Pillow ImageGrab"
        except Exception:
            pass

    if os.name == "nt":                          # pragma: no cover - Windows
        # PowerShell single-quoted literals escape a quote by doubling it.
        save_path = str(path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;"
            "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height;"
            "$g=[System.Drawing.Graphics]::FromImage($bmp);"
            "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,"
            "$b.Size);"
            f"$bmp.Save('{save_path}',[System.Drawing.Imaging.ImageFormat]::Png)"
        )
        ok, out = _run(["powershell", "-NoProfile", "-Command", script])
        if ok and path.exists():
            return True, "PowerShell CopyFromScreen"
        return False, ("no capture backend. Install Pillow "
                       "(pip install pedacito-local[vision]) or use Windows "
                       "screenshot tools. " + out[:200])

    if shutil.which("screencapture"):            # macOS
        ok, out = _run(["screencapture", "-x", str(path)])
        if ok and path.exists():
            return True, "screencapture"
        return False, out or "screencapture failed"

    cmds = []
    if shutil.which("grim"):
        cmds.append(["grim", str(path)])
    if shutil.which("spectacle"):
        cmds.append(["spectacle", "-b", "-n", "-o", str(path)])
    if shutil.which("gnome-screenshot"):
        cmds.append(["gnome-screenshot", "-f", str(path)])
    if shutil.which("scrot"):
        cmds.append(["scrot", "-o", str(path)])
    if shutil.which("import"):
        cmds.append(["import", "-window", "root", str(path)])
    for cmd in cmds:
        ok, out = _run(cmd)
        if ok and path.exists():
            return True, Path(cmd[0]).name
    return False, ("no screenshot backend found. Install Pillow "
                   "(pip install pedacito-local[vision]), grim (Wayland), "
                   "scrot/ImageMagick (X11), or gnome-screenshot.")


_TESSERACT_INIT_ERRORS = (
    "error opening data file",
    "failed loading language",
    "tesseract couldn't load",
    "could not initialize tesseract",
    "please make sure the tessdata_prefix environment variable is set",
)


def ocr(path) -> str:
    """Return OCR text for `path` when tesseract is installed, else ''.

    Without language data, tesseract's stderr reads like OCR output; drop
    those initialisation lines and report no text if nothing real remains.
    """
    if not shutil.which("tesseract"):
        return ""
    ok, out = _run(["tesseract", str(path), "stdout"], timeout=30)
    lines = [ln for ln in (out or "").splitlines()
             if not any(m in ln.lower() for m in _TESSERACT_INIT_ERRORS)]
    text = "\n".join(lines).strip()
    if not ok or not text:
        return ""
    return text[:2000]


def _compare(Image, path: Path, previous: Path) -> str:
    try:
        a = Image.open(path).convert("RGB").resize((160, 90))
        b = Image.open(previous).convert("RGB").resize((160, 90))
    except Exception:
        return ""
    ra, rb = a.tobytes(), b.tobytes()
    pa = list(zip(ra[0::3], ra[1::3], ra[2::3]))
    pb = list(zip(rb[0::3], rb[1::3], rb[2::3]))
    total = 0
    changed = 0
    for (r1, g1, b1), (r2, g2, b2) in zip(pa, pb):
        d = (abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)) / 3
        total += d
        if d > 16:
            changed += 1
    return (f"difference from previous screenshot: "
            f"{total / max(1, len(pa)):.1f}/255 mean, "
            f"{100 * changed / max(1, len(pa)):.1f}% of pixels changed")


def analyze(path, previous=None) -> str:
    """Return a text report about a screenshot: size, brightness, blankness,
    colours, OCR text, and (when `previous` is given) a rough diff."""
    path = Path(path)
    ocr_text = ocr(path)
    pillow = _pillow()
    if pillow is None:
        msg = ("Screenshot saved, but Pillow is not installed, so pixel "
               "analysis is unavailable (pip install pedacito-local[vision]).")
        if ocr_text:
            msg += "\n\nOCR text:\n" + ocr_text
        return msg
    Image, _ = pillow
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return f"Could not read screenshot: {e}"

    w, h = img.size
    small = img.resize((160, 90))
    raw = small.tobytes()
    pixels = list(zip(raw[0::3], raw[1::3], raw[2::3]))
    lums = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pixels]
    n = max(1, len(lums))
    mean = sum(lums) / n
    dark = sum(1 for v in lums if v < 12) / n
    bright = sum(1 for v in lums if v > 243) / n
    colours = len({(r // 16, g // 16, b // 16) for r, g, b in pixels})

    if mean < 8 and dark > 0.95:
        verdict = ("essentially BLACK/BLANK: almost every pixel is near-black "
                   "-- nothing appears to have rendered")
    elif bright > 0.95:
        verdict = ("essentially WHITE/BLANK: almost every pixel is near-white "
                   "-- the window may not have drawn")
    elif colours <= 2:
        verdict = ("nearly UNIFORM color: very little visual content")
    else:
        verdict = "has visible variation (not a uniform blank frame)"

    lines = [
        f"size: {w}x{h}",
        f"mean luminance: {mean:.1f}/255",
        f"near-black pixels: {100 * dark:.1f}%",
        f"near-white pixels: {100 * bright:.1f}%",
        f"distinct quantized colors: {colours}",
        f"verdict: {verdict}",
    ]
    if previous:
        diff = _compare(Image, path, Path(previous))
        if diff:
            lines.append(diff)
    if ocr_text:
        lines.append("OCR text:\n" + ocr_text)
    else:
        lines.append("OCR text: (none detected or tesseract not installed)")
    return "\n".join(lines)


VISION_PROMPT = """\
You are inspecting a screenshot of a running application to help a coding
agent diagnose it. Answer concretely from the image only:
1. Is the window blank, black, white, or visibly rendering content?
2. Is there any visible error, dialog, menu, or loading screen?
3. What is actually visible (describe the main shapes/text/UI), in a few
   sentences?
Do not speculate about code you cannot see in the image. Be concise.
"""


def describe(client, cfg, path) -> str:
    """Ask the configured multimodal model to describe `path`; '' if unset."""
    model = (getattr(cfg, "vision_model", "") or "").strip()
    if not model or client is None or not getattr(client, "vision", None):
        return ""
    return client.vision(
        VISION_PROMPT, path, model=model,
        max_tokens=int(getattr(cfg, "vision_max_tokens", 400) or 400))
