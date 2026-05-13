"""
Apply Nature/Science-style 2D labels onto the clean Blender render.

The Blender script has already projected each label's anchor (3D centre of
the optical component) AND its label position (3D offset above/beside the
component) to 2D pixel coordinates. This script simply draws a thin pointer
line from anchor_px → label_px and places sans-serif text at label_px.

Reads:
    optical_path_clean.png
    optical_path_anchors.json
Writes:
    optical_path_render.png
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN   = os.path.join(OUT_DIR, "optical_path_clean.png")
JSON_IN = os.path.join(OUT_DIR, "optical_path_anchors.json")
FINAL   = os.path.join(OUT_DIR, "optical_path_render.png")

img  = Image.open(CLEAN).convert("RGBA")
draw = ImageDraw.Draw(img, "RGBA")
W, H = img.size

with open(JSON_IN) as f:
    anchors = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Fonts
# ─────────────────────────────────────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def load_font(size, bold=False):
    for p in FONT_PATHS:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".ttc"):
                return ImageFont.truetype(p, size, index=4 if bold else 0)
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


scale = H / 1200
FS_MAIN = int(28 * scale)
FS_SUB  = int(20 * scale)
font_main = load_font(FS_MAIN, bold=True)
font_sub  = load_font(FS_SUB)


# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
TEXT_PRIMARY = "#1f2937"
TEXT_GREEN   = "#1F8478"
TEXT_RED     = "#C62E3C"
TEXT_SUB     = "#6B7280"
LEADER       = (75, 85, 99, 230)
DOT_COLOR    = "#1f2937"


# ─────────────────────────────────────────────────────────────────────────────
# Label specifications
#   (anchor_key, main_text, sub_text, color, h_align)
# h_align controls horizontal alignment of the text relative to label_px
# (its 3D-projected position): "center", "left", or "right"
# ─────────────────────────────────────────────────────────────────────────────
LABELS = [
    ("green_laser",   "Green Laser",       "520 nm",
     TEXT_GREEN,   "center"),
    ("beam_expander", "Beam Expander",     "",
     TEXT_PRIMARY, "center"),
    ("collimating_lens", "Collimating Lens", "",
     TEXT_PRIMARY, "center"),
    ("slm",           "SLM",               "Spatial Light Modulator",
     TEXT_PRIMARY, "center"),
    ("focus_lens",    "Focusing Lens",     "",
     TEXT_PRIMARY, "center"),
    ("side_polish",   "Side-polished",     "region",
     TEXT_PRIMARY, "center"),
    ("mm_fiber",      "Multi-mode Fiber",  "(9 cm rigid rod)",
     TEXT_PRIMARY, "center"),
    ("left_endface",  "Left End-Face",     "",
     TEXT_PRIMARY, "right"),
    ("right_endface", "Right End-Face",    "",
     TEXT_PRIMARY, "left"),
    ("camera",        "CMOS Camera",       "",
     TEXT_PRIMARY, "left"),
    ("red_laser",     "Red Laser",         "650 nm",
     TEXT_RED,     "center"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
LEADER_WIDTH = max(2, int(2.0 * scale))
DOT_R        = max(4, int(5   * scale))
GAP_LEAD     = int(14 * scale)


def text_size(s, font):
    bbox = draw.textbbox((0, 0), s, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# Stage 1: collect projected pixel coords + sizes
items = []
for key, main, sub, color, h_align in LABELS:
    a = anchors.get(key)
    if not a:
        continue
    apx, apy = a["anchor_px"]
    lpx, lpy = a["label_px"]
    wm, hm = text_size(main, font_main)
    if sub:
        ws, hs = text_size(sub, font_sub)
    else:
        ws, hs = 0, 0
    items.append({
        "key": key, "main": main, "sub": sub, "color": color, "h_align": h_align,
        "ax": apx, "ay": apy,
        "lx": lpx, "ly": lpy,
        "wm": wm, "hm": hm, "ws": ws, "hs": hs,
        "box_w": max(wm, ws), "box_h": hm + (hs + int(4 * scale) if sub else 0),
    })


# Stage 2: lightweight canvas clamping — only reposition labels that
# would project completely outside the canvas. Otherwise trust the 3D
# offsets and let labels float wherever the projection puts them.
PAD = int(20 * scale)
for it in items:
    box_w, box_h = it["box_w"], it["box_h"]
    half_w = box_w / 2
    if it["h_align"] == "center":
        x_min = PAD + half_w
        x_max = W - PAD - half_w
    elif it["h_align"] == "left":
        x_min = PAD
        x_max = W - PAD - box_w
    else:  # right
        x_min = PAD + box_w
        x_max = W - PAD
    it["lx"] = max(x_min, min(x_max, it["lx"]))
    y_min = PAD + it["hm"] / 2
    y_max = H - PAD - box_h
    it["ly"] = max(y_min, min(y_max, it["ly"]))


# Stage 3: render leader lines first (under the text)
for it in items:
    ax, ay = int(it["ax"]), int(it["ay"])
    lx, ly = int(it["lx"]), int(it["ly"])
    h = int(it["hm"] / 2)

    # Connector end-point: just outside the text box, on the side facing the anchor
    if it["h_align"] == "center":
        end_x = lx
    elif it["h_align"] == "left":
        end_x = lx if ax < lx else lx + it["wm"]
    else:  # right
        end_x = lx if ax > lx else lx - it["wm"]

    # The text baseline is roughly at ly + hm; line should connect to its top
    # if anchor is above, or bottom if anchor is below.
    if ay < ly:                        # anchor above label → line ends at label top
        end_y = ly - GAP_LEAD
    elif ay > ly + it["hm"]:           # anchor below label → line ends at label bottom
        end_y = ly + it["hm"] + GAP_LEAD
    else:                              # anchor sideways → connect to mid-line
        end_y = ly + h

    # Draw line
    draw.line([(ax, ay), (end_x, end_y)],
              fill=LEADER, width=LEADER_WIDTH, joint="curve")

    # Anchor dot
    draw.ellipse([(ax - DOT_R, ay - DOT_R), (ax + DOT_R, ay + DOT_R)],
                 fill=DOT_COLOR)


# Stage 4: render text on top
for it in items:
    lx, ly = int(it["lx"]), int(it["ly"])
    if it["h_align"] == "center":
        x_main = int(lx - it["wm"] / 2)
        x_sub  = int(lx - it["ws"] / 2) if it["sub"] else x_main
    elif it["h_align"] == "left":
        x_main = lx
        x_sub  = lx
    else:  # right
        x_main = lx - it["wm"]
        x_sub  = lx - it["ws"] if it["sub"] else x_main

    # Optional subtle white outline so labels remain readable when crossing
    # a beam edge by 1–2 px.
    for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x_main + ox, ly + oy), it["main"],
                  fill=(255, 255, 255, 200), font=font_main)
    draw.text((x_main, ly), it["main"], fill=it["color"], font=font_main)

    if it["sub"]:
        sy = ly + it["hm"] + int(4 * scale)
        for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
            draw.text((x_sub + ox, sy + oy), it["sub"],
                      fill=(255, 255, 255, 200), font=font_sub)
        draw.text((x_sub, sy), it["sub"], fill=TEXT_SUB, font=font_sub)


# ─────────────────────────────────────────────────────────────────────────────
# Channel legend (bottom-right corner)
# ─────────────────────────────────────────────────────────────────────────────
lgd_main  = load_font(int(20 * scale), bold=False)
swatch_w  = int(40 * scale)
swatch_h  = int(8  * scale)
lgd_x = W - int(440 * scale)
lgd_y = H - int(160 * scale)

draw.rounded_rectangle(
    [(lgd_x, lgd_y), (lgd_x + swatch_w, lgd_y + swatch_h)],
    radius=int(2 * scale), fill="#2A9D8F")
draw.text((lgd_x + swatch_w + int(14*scale), lgd_y - int(10*scale)),
          "Green channel  (signal, side illumination)",
          fill=TEXT_PRIMARY, font=lgd_main)
draw.rounded_rectangle(
    [(lgd_x, lgd_y + int(46 * scale)),
     (lgd_x + swatch_w, lgd_y + int(46 * scale) + swatch_h)],
    radius=int(2 * scale), fill="#E63946")
draw.text((lgd_x + swatch_w + int(14*scale), lgd_y + int(36 * scale)),
          "Red channel  (reference, end-face axial)",
          fill=TEXT_PRIMARY, font=lgd_main)


img.convert("RGB").save(FINAL, "PNG", optimize=True)
print("Final figure saved to:", FINAL)
