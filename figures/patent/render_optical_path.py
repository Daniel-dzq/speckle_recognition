"""
Nature/Science journal-style optical path render — physically correct edition.

Topology (rewritten per the experiment description):
    * Multi-mode fiber  — straight 9 cm cylinder along the +X axis at y=z=0.
                          Has well-defined Left End-Face, Right End-Face, and
                          a Side-polished region in the middle.
    * Red channel       — Red laser shoots straight along the fiber axis
                          (coaxial), directly into the Left End-Face.
                          NO SM fiber coupler.
    * Green channel     — Green Laser → Beam Expander → SLM → Focusing Lens
                          all on a single straight optical axis tilted ~35°
                          relative to the fiber axis. Beam intersects the
                          fiber's SIDE at the side-polished region.
    * CMOS Camera       — Sits exactly at the Right End-Face of the fiber.

Label anchoring strategy (CRITICAL):
    Every label has TWO 3D coordinates:
        anchor_3d  — point on the actual component (centre of mass)
        label_3d   — anchor_3d  +  per-component offset vector
    Both are projected to 2D pixel space at render time and saved to
    optical_path_anchors.json. The labeller then draws each pointer line
    from anchor_px → label_px and places text at label_px.

Run:
    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python figures/patent/render_optical_path.py
"""
import bpy
import os
import math
import json
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


# ─────────────────────────────────────────────────────────────────────────────
# 0. Reset & basic settings
# ─────────────────────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.render.resolution_x = 4000
scene.render.resolution_y = 2200
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
PNG_OUT   = os.path.join(OUT_DIR, "optical_path_clean.png")
BLEND_OUT = os.path.join(OUT_DIR, "optical_path_scene.blend")
JSON_OUT  = os.path.join(OUT_DIR, "optical_path_anchors.json")
scene.render.filepath = PNG_OUT
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.compression = 15


# ─────────────────────────────────────────────────────────────────────────────
# 1. White world
# ─────────────────────────────────────────────────────────────────────────────
world = bpy.data.worlds.new("PUFWorld")
scene.world = world
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()
n_out = nt.nodes.new("ShaderNodeOutputWorld")
n_bg  = nt.nodes.new("ShaderNodeBackground")
n_bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
n_bg.inputs["Strength"].default_value = 1.05
nt.links.new(n_bg.outputs["Background"], n_out.inputs["Surface"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Materials
# ─────────────────────────────────────────────────────────────────────────────
def hex_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def mat_principled(name, base, metallic=0.0, roughness=0.5, ior=1.45,
                   transmission=0.0, alpha=1.0, blend="OPAQUE"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission
    bsdf.inputs["Alpha"].default_value = alpha
    if blend == "BLEND":
        mat.blend_method = "BLEND"
    return mat


CY_GLASS   = hex_rgb("88CCEE")
GREEN_BEAM = hex_rgb("10FF50")     # neon laser green
RED_BEAM   = hex_rgb("FF1515")     # vibrant laser red
SLATE_DARK = hex_rgb("454D55")
SILVER     = hex_rgb("D3D9DF")
GOLD       = hex_rgb("E0B36A")

M_GLASS_FIBER = mat_principled("glass_fiber", CY_GLASS,
                               metallic=0.0, roughness=0.10,
                               transmission=0.6, alpha=0.4, blend="BLEND",
                               ior=1.46)
M_GLASS_LENS  = mat_principled("glass_lens", CY_GLASS,
                               metallic=0.0, roughness=0.05,
                               transmission=0.85, ior=1.50)
M_SLATE       = mat_principled("slate_matte", SLATE_DARK,
                               metallic=0.0, roughness=0.55)
M_SILVER      = mat_principled("silver_metal", SILVER,
                               metallic=0.95, roughness=0.30)
M_GOLD        = mat_principled("gold_accent", GOLD,
                               metallic=1.0, roughness=0.25)
M_BEAM_GREEN  = mat_principled("beam_green", GREEN_BEAM,
                               metallic=0.0, roughness=0.40,
                               alpha=0.5, blend="BLEND")
M_BEAM_RED    = mat_principled("beam_red", RED_BEAM,
                               metallic=0.0, roughness=0.40,
                               alpha=0.5, blend="BLEND")
M_LETTER      = mat_principled("letter_face", hex_rgb("FFFFFF"),
                               metallic=0.0, roughness=0.30)
M_END_FACE    = mat_principled("end_face", CY_GLASS,
                               metallic=0.1, roughness=0.05,
                               transmission=0.3, alpha=0.85, blend="BLEND")
M_POLISH      = mat_principled("polish", hex_rgb("9AA3AF"),
                               metallic=0.90, roughness=0.18)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def add_cylinder(name, radius, depth, location, rotation=(0,0,0),
                 material=None, segments=64):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, location=location,
        rotation=rotation, vertices=segments)
    o = bpy.context.object; o.name = name
    bpy.ops.object.shade_smooth()
    if material: o.data.materials.append(material)
    return o


def add_cube(name, size, location, rotation=(0,0,0), scale=(1,1,1),
             material=None):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location,
                                     rotation=rotation)
    o = bpy.context.object; o.name = name
    o.scale = scale
    if material: o.data.materials.append(material)
    return o


def bevel(obj, width=0.02, segments=4):
    mod = obj.modifiers.new("Bevel", "BEVEL")
    mod.width = width; mod.segments = segments; mod.limit_method = "ANGLE"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def beam_segment(name, p0, p1, radius, material):
    """Cylinder beam from p0 to p1 (constant radius)."""
    p0v, p1v = Vector(p0), Vector(p1)
    direction = p1v - p0v
    length = direction.length
    if length == 0: return None
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length, location=(p0v + p1v) / 2, vertices=48)
    o = bpy.context.object; o.name = name
    z = Vector((0, 0, 1))
    rot = z.rotation_difference(direction.normalized()).to_euler()
    o.rotation_euler = rot
    if material: o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o


def cone_segment(name, p0, p1, radius_start, radius_end, material):
    """Truncated cone (frustum) from p0 (radius_start) to p1 (radius_end)."""
    p0v, p1v = Vector(p0), Vector(p1)
    direction = p1v - p0v
    length = direction.length
    if length == 0: return None
    bpy.ops.mesh.primitive_cone_add(
        radius1=radius_start, radius2=radius_end,
        depth=length, location=(p0v + p1v) / 2, vertices=64)
    o = bpy.context.object; o.name = name
    z = Vector((0, 0, 1))
    rot = z.rotation_difference(direction.normalized()).to_euler()
    o.rotation_euler = rot
    if material: o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o


# ─────────────────────────────────────────────────────────────────────────────
# 4. Layout — fiber as a straight slanted cylinder
# ─────────────────────────────────────────────────────────────────────────────
# Fiber is laid along +X axis. Length 9 (= 9 cm). Slight upward tilt at
# the right end so it reads as 3D in iso view but stays straight.
FIBER_LEFT  = Vector((-2.0, 0.0, 0.05))
FIBER_RIGHT = Vector(( 7.0, 0.0, -0.05))
FIBER_AXIS  = (FIBER_RIGHT - FIBER_LEFT).normalized()
FIBER_MID   = (FIBER_LEFT + FIBER_RIGHT) / 2.0
FIBER_RADIUS = 0.22
FIBER_LENGTH = (FIBER_RIGHT - FIBER_LEFT).length

# Build the fiber cylinder along its own length, then orient via Z-axis trick
fiber = add_cylinder("MMFiber",
                     radius=FIBER_RADIUS,
                     depth=FIBER_LENGTH,
                     location=tuple(FIBER_MID),
                     material=M_GLASS_FIBER, segments=96)
# Rotate so its central axis aligns with FIBER_AXIS
z_unit = Vector((0, 0, 1))
fiber.rotation_euler = z_unit.rotation_difference(FIBER_AXIS).to_euler()


# ── End-face discs (slightly outset so they're visible) ────────────────────
def end_face_disc(name, centre, radius, material, axis_in=Vector((1,0,0))):
    """Create a thin disc at `centre`, normal aligned with `axis_in`."""
    disc = add_cylinder(name, radius=radius, depth=0.012,
                        location=tuple(centre),
                        material=material)
    disc.rotation_euler = z_unit.rotation_difference(axis_in.normalized()
                                                     ).to_euler()
    return disc

end_face_disc("LeftEndFace",  FIBER_LEFT  + 0.005 * FIBER_AXIS,
              FIBER_RADIUS * 1.02, M_END_FACE,  axis_in=FIBER_AXIS)
end_face_disc("RightEndFace", FIBER_RIGHT - 0.005 * FIBER_AXIS,
              FIBER_RADIUS * 1.02, M_END_FACE,  axis_in=FIBER_AXIS)


# ── Side-polished region (flat patch on top of the fiber, mid-section) ─────
SP_t = 0.50  # parametric position (0=left, 1=right)
SP_centre = FIBER_LEFT + (FIBER_RIGHT - FIBER_LEFT) * SP_t
SP_topward = Vector((0, 0, FIBER_RADIUS + 0.005))   # straight up from the axis
SP_top_centre = SP_centre + SP_topward

polished = add_cube("PolishedRegion", 1.0,
                    location=tuple(SP_top_centre),
                    scale=(0.65, 0.20, 0.018),
                    material=M_POLISH)
bevel(polished, width=0.005, segments=3)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Red channel — coaxial with the fiber, shoots into Left End-Face
# ─────────────────────────────────────────────────────────────────────────────
# Place red laser BEHIND the Left End-Face along -X axis (i.e., -FIBER_AXIS).
LASER_RED_BACK = FIBER_LEFT - 2.40 * FIBER_AXIS   # back of laser body
LASER_RED_FRONT = FIBER_LEFT - 1.10 * FIBER_AXIS   # front of laser body
LASER_RED_HEAD  = FIBER_LEFT - 0.95 * FIBER_AXIS   # gold head centre

red_body_centre = (LASER_RED_BACK + LASER_RED_FRONT) / 2
red_body_len    = (LASER_RED_FRONT - LASER_RED_BACK).length
r_body = add_cylinder("RedLaser_Body",
                      radius=0.30,
                      depth=red_body_len,
                      location=tuple(red_body_centre),
                      material=M_SILVER)
r_body.rotation_euler = z_unit.rotation_difference(FIBER_AXIS).to_euler()
bevel(r_body, width=0.04, segments=3)

r_head = add_cylinder("RedLaser_Head",
                      radius=0.20, depth=0.18,
                      location=tuple(LASER_RED_HEAD),
                      material=M_GOLD)
r_head.rotation_euler = z_unit.rotation_difference(FIBER_AXIS).to_euler()
bevel(r_head, width=0.018, segments=3)

# Red beam: coaxial, from head to Left End-Face
beam_segment("BeamR_axial",
             tuple(LASER_RED_HEAD + 0.10 * FIBER_AXIS),
             tuple(FIBER_LEFT - 0.04 * FIBER_AXIS),
             radius=0.06, material=M_BEAM_RED)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Green channel — straight axis, hits the side-polished region at ~35°
# ─────────────────────────────────────────────────────────────────────────────
# Direction FROM the green source TO the side-polished hit point.
# Decomposed so that the angle with the fiber axis is 35°.
THETA = math.radians(35)
# Beam direction has +X (down the fiber), -Y (toward viewer), -Z (downward)
# components of magnitude that sums to 1 at the chosen angle.
gx = math.cos(THETA)            # component along +X (fiber axis)
gp = math.sin(THETA)            # perpendicular magnitude
# Distribute perpendicular component between Y (toward viewer) and Z (down).
# Choose mostly -Z (light coming from above) with a touch of -Y.
GREEN_DIR = Vector((gx, -0.55 * gp, -0.83 * gp)).normalized()

# Hit point: top of the side-polished region.
HIT = SP_top_centre.copy()

# Place components along the green axis (parametric distance from HIT).
def green_pos(t):
    """Return world coord at parameter t along the green optical axis.

    GREEN_DIR points FROM the source TO the hit. So:
        t = 0   → hit point
        t < 0  → upstream toward the laser source
        t > 0  → downstream past the fiber (unused)
    """
    return HIT + t * GREEN_DIR


# Component positions along the axis (parameter t, world units along GREEN_DIR)
# t < 0 is upstream (laser side); t = 0 is the hit point on the fiber.
T_LASER_FRONT  = -4.5
T_LASER_BACK   = -5.8
T_EXPANDER     = -3.6
T_COLLIMATOR   = -2.5     # NEW: collimating lens between expander and SLM
T_SLM          = -1.8
T_FOC_LENS     = -1.0
T_HIT          =  0.0

# Rotation for green-axis components: their cylinder axis must align with
# GREEN_DIR (so light flows along it).
GREEN_ROT = z_unit.rotation_difference(GREEN_DIR).to_euler()

# Green laser body
g_body_centre = (green_pos(T_LASER_FRONT) + green_pos(T_LASER_BACK)) / 2
g_body_len    = (green_pos(T_LASER_FRONT) - green_pos(T_LASER_BACK)).length
g_body = add_cylinder("GreenLaser_Body",
                      radius=0.30,
                      depth=g_body_len,
                      location=tuple(g_body_centre),
                      material=M_SILVER)
g_body.rotation_euler = GREEN_ROT
bevel(g_body, width=0.04, segments=3)

# Green laser gold head
g_head_pos = green_pos(T_LASER_FRONT) - 0.10 * GREEN_DIR
g_head = add_cylinder("GreenLaser_Head",
                      radius=0.22, depth=0.20,
                      location=tuple(g_head_pos),
                      material=M_GOLD)
g_head.rotation_euler = GREEN_ROT
bevel(g_head, width=0.02, segments=3)

# Beam expander (small diverging lens — narrow-input ⇒ widening output)
beam_exp = add_cylinder("BeamExpander",
                        radius=0.18, depth=0.08,
                        location=tuple(green_pos(T_EXPANDER)),
                        material=M_GLASS_LENS)
beam_exp.rotation_euler = GREEN_ROT

# Collimating lens (large lens that captures the diverging beam and produces
# a parallel collimated wave — placed between expander and SLM)
collim_lens = add_cylinder("CollimatingLens",
                           radius=0.42, depth=0.12,
                           location=tuple(green_pos(T_COLLIMATOR)),
                           material=M_GLASS_LENS)
collim_lens.rotation_euler = GREEN_ROT
collim_ring = add_cylinder("CollimatingLens_Ring",
                           radius=0.47, depth=0.16,
                           location=tuple(green_pos(T_COLLIMATOR)),
                           material=M_SLATE)
collim_ring.rotation_euler = GREEN_ROT

# SLM panel — face perpendicular to the optical axis
slm_pos = green_pos(T_SLM)
slm = add_cube("SLM", 1.0,
               location=tuple(slm_pos),
               scale=(0.05, 0.75, 0.75),
               material=M_SLATE)
# rotate SLM so its thin axis (X-local) aligns with GREEN_DIR
slm.rotation_euler = z_unit.rotation_difference(GREEN_DIR).to_euler()
# add a quarter-turn so its face is perpendicular to the optical axis
import math as _m
slm.rotation_euler.rotate_axis("Y", _m.radians(90))
bevel(slm, width=0.015, segments=3)

# Letter "A" on the SLM front face (facing back toward the laser)
letter_curve = bpy.data.curves.new("LetterA", "FONT")
letter_curve.body = "A"
letter_curve.size = 0.55
letter_curve.extrude = 0.005
letter_curve.align_x = "CENTER"
letter_curve.align_y = "CENTER"
letter_obj = bpy.data.objects.new("LetterA", letter_curve)
bpy.context.collection.objects.link(letter_obj)
letter_offset = -0.04 * GREEN_DIR
letter_obj.location = tuple(slm_pos + letter_offset)
letter_obj.rotation_euler = slm.rotation_euler.copy()
letter_obj.data.materials.append(M_LETTER)

# Focusing lens
foc = add_cylinder("FocusLens",
                   radius=0.45, depth=0.12,
                   location=tuple(green_pos(T_FOC_LENS)),
                   material=M_GLASS_LENS)
foc.rotation_euler = GREEN_ROT
foc_ring = add_cylinder("FocusLens_Ring",
                        radius=0.50, depth=0.16,
                        location=tuple(green_pos(T_FOC_LENS)),
                        material=M_SLATE)
foc_ring.rotation_euler = GREEN_ROT

# ── Green beam — four physically distinct geometric segments ───────────────
#   1. Raw narrow beam      : laser head     → beam expander
#   2. Diverging frustum    : beam expander  → collimating lens   (cone open)
#   3. Wide collimated beam : collimating L. → focusing lens      (parallel, through SLM)
#   4. Converging frustum   : focusing lens  → side-polished hit  (cone close)
RAW_RADIUS  = 0.045    # narrow incoming beam (~1-2 mm physical)
WIDE_RADIUS = 0.260    # collimated expanded beam (covers SLM aperture)
HIT_RADIUS  = 0.025    # focused spot at the side-polished region

# Segment 1 — raw narrow cylinder
beam_segment("BeamG1_raw",
             tuple(g_head_pos),
             tuple(green_pos(T_EXPANDER) - 0.04 * GREEN_DIR),
             radius=RAW_RADIUS, material=M_BEAM_GREEN)

# Segment 2 — diverging cone after the beam expander
cone_segment("BeamG2_diverging",
             tuple(green_pos(T_EXPANDER)   + 0.04 * GREEN_DIR),
             tuple(green_pos(T_COLLIMATOR) - 0.06 * GREEN_DIR),
             radius_start=RAW_RADIUS,
             radius_end=WIDE_RADIUS,
             material=M_BEAM_GREEN)

# Segment 3 — wide collimated parallel cylinder (passes through the SLM)
beam_segment("BeamG3_collimated",
             tuple(green_pos(T_COLLIMATOR) + 0.06 * GREEN_DIR),
             tuple(green_pos(T_FOC_LENS)   - 0.06 * GREEN_DIR),
             radius=WIDE_RADIUS, material=M_BEAM_GREEN)

# Segment 4 — converging cone (wide → tiny) ending at the fiber side
cone_segment("BeamG4_focusing",
             tuple(green_pos(T_FOC_LENS) + 0.06 * GREEN_DIR),
             tuple(HIT + 0.04 * GREEN_DIR),
             radius_start=WIDE_RADIUS,
             radius_end=HIT_RADIUS,
             material=M_BEAM_GREEN)

# Bright coupling spot
bpy.ops.mesh.primitive_uv_sphere_add(
    radius=0.10, location=tuple(HIT + Vector((0, 0, 0.005))),
    segments=32, ring_count=24)
spot = bpy.context.object; spot.name = "GreenCouplingSpot"
spot.data.materials.append(
    mat_principled("coupling_spot", hex_rgb("CFFAF0"),
                   metallic=0.0, roughness=0.30,
                   alpha=0.85, blend="BLEND"))
bpy.ops.object.shade_smooth()


# ─────────────────────────────────────────────────────────────────────────────
# 7. CMOS camera — at the Right End-Face
# ─────────────────────────────────────────────────────────────────────────────
CAM_LENS_CENTRE = FIBER_RIGHT + 0.30 * FIBER_AXIS
CAM_BODY_CENTRE = FIBER_RIGHT + 0.85 * FIBER_AXIS

cam_lens = add_cylinder("CMOSLens",
                        radius=0.22, depth=0.30,
                        location=tuple(CAM_LENS_CENTRE),
                        material=M_SILVER)
cam_lens.rotation_euler = z_unit.rotation_difference(FIBER_AXIS).to_euler()
bevel(cam_lens, width=0.02, segments=3)

cam_glass = add_cylinder("CMOSGlass",
                         radius=0.16, depth=0.05,
                         location=tuple(FIBER_RIGHT + 0.165 * FIBER_AXIS),
                         material=M_GLASS_LENS)
cam_glass.rotation_euler = z_unit.rotation_difference(FIBER_AXIS).to_euler()

cam_body = add_cube("CMOSBody", 1.0,
                    location=tuple(CAM_BODY_CENTRE),
                    scale=(0.50, 0.50, 0.50),
                    material=M_SLATE)
# rotate so its 'front' faces the fiber
cam_body.rotation_euler = z_unit.rotation_difference(FIBER_AXIS).to_euler()
bevel(cam_body, width=0.04, segments=3)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Anchors: 3D coords for pointer-line start AND label position
# ─────────────────────────────────────────────────────────────────────────────
# For each label we store (anchor_3d, label_3d).
#   anchor_3d   — exact centre of the component being labelled.
#   label_3d    — where the text is drawn; chosen as anchor_3d + offset_3d
#                 so labels float in clean positions and never overlap
#                 the optical components or beams.
# Re-recompute key positions now that green_pos is fixed.
g_body_centre = (green_pos(T_LASER_FRONT) + green_pos(T_LASER_BACK)) / 2

# Offsets are 3D vectors added to each anchor to position the label tag.
# We stagger directions so that, when projected to 2D, labels never sit on
# top of each other and never cross the optical components or beams.
# Big 3D Z offsets project to large 2D Y offsets in iso view.
# Top labels: +Z ≈ 3.0 ; bottom labels: -Z ≈ 3.0
# Lateral spread uses ±X to avoid 2D collisions of components that share
# similar X (e.g. mm_fiber and side_polish are both at X=2.5).
ANCHORS = {
    # ── TOP HALF of canvas ─────────────────────────────────────────
    "green_laser":   {
        "anchor": tuple(g_body_centre),
        "label":  tuple(g_body_centre + Vector((-1.0, 0.0, 0.6))),
    },
    "beam_expander": {
        "anchor": tuple(green_pos(T_EXPANDER)),
        "label":  tuple(green_pos(T_EXPANDER) + Vector((-1.8, 0.0, 0.5))),
    },
    "collimating_lens": {
        "anchor": tuple(green_pos(T_COLLIMATOR)),
        "label":  tuple(green_pos(T_COLLIMATOR) + Vector((-1.0, 0.0, 1.0))),
    },
    "slm": {
        "anchor": tuple(slm_pos),
        "label":  tuple(slm_pos + Vector((-0.6, 0.0, 1.4))),
    },
    "focus_lens": {
        "anchor": tuple(green_pos(T_FOC_LENS)),
        "label":  tuple(green_pos(T_FOC_LENS) + Vector((1.0, 0.0, 1.6))),
    },
    "side_polish": {
        "anchor": tuple(SP_top_centre),
        "label":  tuple(SP_top_centre + Vector((2.0, 0.0, 1.5))),
    },
    "camera": {
        "anchor": tuple(CAM_BODY_CENTRE),
        "label":  tuple(CAM_BODY_CENTRE + Vector((0.6, 0.0, 1.6))),
    },

    # ── BOTTOM HALF of canvas ──────────────────────────────────────
    "red_laser": {
        "anchor": tuple(red_body_centre),
        "label":  tuple(red_body_centre + Vector((-1.0, 0.0, -1.4))),
    },
    "left_endface": {
        "anchor": tuple(FIBER_LEFT),
        "label":  tuple(FIBER_LEFT + Vector((0.5, 0.0, -1.6))),
    },
    "mm_fiber": {
        "anchor": tuple(FIBER_MID + Vector((0, 0, -FIBER_RADIUS))),
        "label":  tuple(FIBER_MID + Vector((0.0, 0.0, -2.2))),
    },
    "right_endface": {
        "anchor": tuple(FIBER_RIGHT),
        "label":  tuple(FIBER_RIGHT + Vector((-0.4, 0.0, -1.5))),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Lighting — clean academic
# ─────────────────────────────────────────────────────────────────────────────
def add_area_light(name, location, rotation, energy, color=(1,1,1), size=4.0):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy; ld.color = color; ld.size = size
    obj = bpy.data.objects.new(name, ld)
    bpy.context.collection.objects.link(obj)
    obj.location = location; obj.rotation_euler = rotation
    return obj


add_area_light("KeyTop", (3.0, 0.0, 8.0), (0, 0, 0), 700,
               color=(1.0, 1.0, 1.0), size=14.0)
add_area_light("FrontFill", (3.0, -8.0, 4.0),
               (math.radians(60), 0, 0), 280,
               color=(0.95, 0.97, 1.0), size=10.0)
add_area_light("BottomBounce", (3.0, 0.0, -4.0),
               (math.radians(180), 0, 0), 80,
               color=(1.0, 1.0, 1.0), size=12.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Camera — orthographic iso (35° elev, -50° az)
#
# Manually build the camera-to-world matrix so that:
#   * camera local -Z (forward)   →  look_dir  (camera looks at target)
#   * camera local +Y (frame up)  →  world +Z  (so world Z = vertical screen)
# ─────────────────────────────────────────────────────────────────────────────
import mathutils

target = Vector((2.5, 0.0, 0.6))
elevation = math.radians(35)
azimuth   = math.radians(-50)
distance  = 25.0

cam_offset = Vector((
    -math.sin(azimuth) * math.cos(elevation),
    -math.cos(azimuth) * math.cos(elevation),
     math.sin(elevation),
)) * distance
cam_loc = target + cam_offset

cam_data = bpy.data.cameras.new("HeroCamera")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 20.0
cam_obj = bpy.data.objects.new("HeroCamera", cam_data)
bpy.context.collection.objects.link(cam_obj)

# Build world-Z-up look-at matrix manually
forward = (target - cam_loc).normalized()
world_up = Vector((0.0, 0.0, 1.0))
right   = forward.cross(world_up).normalized()
up      = right.cross(forward).normalized()
# Blender camera columns: [right, up, -forward, location]
mat = mathutils.Matrix((
    (right.x,  up.x,  -forward.x,  cam_loc.x),
    (right.y,  up.y,  -forward.y,  cam_loc.y),
    (right.z,  up.z,  -forward.z,  cam_loc.z),
    (0.0,      0.0,    0.0,        1.0),
))
cam_obj.matrix_world = mat
scene.camera = cam_obj


# ─────────────────────────────────────────────────────────────────────────────
# 11. Project anchor + label 3D coords → 2D pixels
# ─────────────────────────────────────────────────────────────────────────────
res_x = scene.render.resolution_x * scene.render.resolution_percentage / 100
res_y = scene.render.resolution_y * scene.render.resolution_percentage / 100


def project_3d(pt3):
    co = world_to_camera_view(scene, cam_obj, Vector(pt3))
    return round(co.x * res_x, 1), round((1.0 - co.y) * res_y, 1)


projected = {}
for name, pair in ANCHORS.items():
    apx, apy = project_3d(pair["anchor"])
    lpx, lpy = project_3d(pair["label"])
    projected[name] = {
        "anchor_px": [apx, apy],
        "label_px":  [lpx, lpy],
        "anchor_world": list(pair["anchor"]),
        "label_world":  list(pair["label"]),
    }
projected["__meta"] = {
    "image_width":  int(res_x),
    "image_height": int(res_y),
}
with open(JSON_OUT, "w") as f:
    json.dump(projected, f, indent=2, ensure_ascii=False)
print("Anchors saved to:", JSON_OUT)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Save .blend then render
# ─────────────────────────────────────────────────────────────────────────────
bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)
print("Scene saved to:", BLEND_OUT)

print("Rendering to:", PNG_OUT)
bpy.ops.render.render(write_still=True)
print("Done.")
