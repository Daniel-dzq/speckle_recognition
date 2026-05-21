"""
Blender 4.x — Fig.2 optical setup (semi-3D infographic).

Run from repo root:
  /Applications/Blender.app/Contents/MacOS/Blender --background \\
      --python scripts/paper_figures/blender_fig2_setup/render_fig2_scene.py -- <REPO_ROOT>

Or: bash scripts/paper_figures/blender_fig2_setup/run_blender_render.sh

Outputs:
  figures/paper/Fig2_setup/Fig2_optical_setup_9cm.blend
  figures/paper/Fig2_setup/Fig2_optical_setup_9cm_render.png
  figures/paper/Fig2_setup/Fig2_optical_setup_9cm_polish_closeup.png

Units: 1 BU = 1 cm. Fiber on X from -4.5 to +4.5.
Polish: x in [0.5, 1.5] (5 + 1 + 3 cm from left).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector

try:
    sep = sys.argv.index("--")
    argv = sys.argv[sep + 1 :]
except ValueError:
    argv = []

REPO_ROOT = Path(argv[0]).resolve() if argv else Path.cwd()
OUT_DIR = REPO_ROOT / "figures" / "paper" / "Fig2_setup"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BLEND_PATH = OUT_DIR / "Fig2_optical_setup_9cm.blend"
MAIN_RENDER = OUT_DIR / "Fig2_optical_setup_9cm_render.png"
CLOSEUP_RENDER = OUT_DIR / "Fig2_optical_setup_9cm_polish_closeup.png"

X_LEFT, X_RIGHT = -4.5, 4.5
X_POL0, X_POL1 = 0.5, 1.5
R_CLAD = 0.26
R_CORE = 0.14
POLISH_Y0 = R_CLAD * 0.98
POLISH_Y1 = R_CLAD + 0.22
POLISH_HALF_Z = 0.35


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def principled_mat(name: str, base_color=(1, 1, 1, 1), metallic=0.0, roughness=0.35, **kw):
    m = bpy.data.materials.new(name=name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (380, 0)
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.location = (0, 0)
    p.inputs["Base Color"].default_value = base_color
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = roughness
    tw = None
    for key in ("Transmission Weight", "Transmission"):
        if key in p.inputs:
            tw = key
            break
    if tw:
        p.inputs[tw].default_value = kw.get("transmission", 0.0)
    if "IOR" in p.inputs:
        p.inputs["IOR"].default_value = kw.get("ior", 1.45)
    if "Alpha" in p.inputs:
        p.inputs["Alpha"].default_value = kw.get("alpha", 1.0)
    if kw.get("emission"):
        p.inputs["Emission Color"].default_value = (*kw["emission_color"][:3], 1.0)
        p.inputs["Emission Strength"].default_value = kw.get("emission_strength", 5.0)
    nt.links.new(p.outputs["BSDF"], out.inputs["Surface"])
    alpha = kw.get("alpha", 1.0)
    if alpha < 0.999 and hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
        m.use_backface_culling = False
    return m


def cyl_along_x(name: str, r: float, x0: float, x1: float, y=0.0, z=0.0, mat=None):
    length = max(x1 - x0, 1e-4)
    cx = (x0 + x1) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64, radius=r, depth=length, location=(cx, y, z), rotation=(0.0, math.pi / 2.0, 0.0)
    )
    ob = bpy.context.active_object
    ob.name = name
    if mat:
        ob.data.materials.append(mat)
    return ob


def cylinder_between(name: str, p0: Vector, p1: Vector, radius: float, mat):
    p0 = Vector(p0)
    p1 = Vector(p1)
    d = p1 - p0
    length = max(d.length, 1e-4)
    loc = (p0 + p1) / 2.0
    quat = d.to_track_quat("Z", "Y")
    rot = quat.to_euler()
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=length, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.materials.append(mat)
    return ob


def box_mesh(name: str, sx, sy, sz, loc, rot_euler=(0, 0, 0), mat=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=tuple(loc), rotation=rot_euler)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        ob.data.materials.append(mat)
    return ob


def add_subsurf(ob, levels=2):
    m = ob.modifiers.new("Subsurf", "SUBSURF")
    m.levels = levels
    m.render_levels = levels


def lens_disk(name: str, center: Vector, axis: Vector, radius=0.42, thick=0.16, mat=None):
    axis = Vector(axis).normalized()
    quat = axis.to_track_quat("Z", "Y")
    rot = quat.to_euler()
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=thick, location=tuple(center), rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    if mat:
        ob.data.materials.append(mat)
    return ob


def setup_world():
    world = bpy.data.worlds.new("Fig2World")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = world


def setup_render_cycles(scene):
    scene.render.engine = "CYCLES"
    scene.render.film_transparent = False
    scene.render.resolution_x = 3200
    scene.render.resolution_y = 2200
    scene.render.resolution_percentage = 100
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 96
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 8
    if hasattr(scene.view_settings, "view_transform"):
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0.45
        scene.view_settings.gamma = 1.0


def main():
    clear_scene()
    scene = bpy.context.scene
    setup_world()
    setup_render_cycles(scene)

    m_clad = principled_mat(
        "MatClad", base_color=(0.93, 0.95, 1.0, 1), roughness=0.06, transmission=0.97, ior=1.49, alpha=0.25
    )
    m_core = principled_mat(
        "MatCore", base_color=(0.5, 0.82, 1.0, 1), roughness=0.05, transmission=0.9, ior=1.48, alpha=0.45
    )
    m_polish = principled_mat(
        "MatPolish",
        base_color=(0.35, 0.92, 0.45, 1),
        roughness=0.2,
        transmission=0.7,
        ior=1.45,
        alpha=0.35,
        emission=True,
        emission_color=(0.25, 0.95, 0.35),
        emission_strength=0.45,
    )
    m_red = principled_mat(
        "MatRedBeam",
        base_color=(1, 0.02, 0.02, 1),
        roughness=1.0,
        emission=True,
        emission_color=(1, 0.15, 0.12),
        emission_strength=38.0,
    )
    m_green = principled_mat(
        "MatGreenBeam",
        base_color=(0.05, 0.95, 0.15, 1),
        roughness=1.0,
        emission=True,
        emission_color=(0.3, 1.0, 0.4),
        emission_strength=22.0,
    )
    m_metal = principled_mat("MatMetal", base_color=(0.1, 0.1, 0.12, 1), metallic=0.95, roughness=0.28)
    m_glass = principled_mat(
        "MatGlass", base_color=(0.97, 0.99, 1.0, 1), roughness=0.02, transmission=0.98, ior=1.52, alpha=0.08
    )
    m_slm = principled_mat("MatSLM", base_color=(0.03, 0.03, 0.04, 1), metallic=0.15, roughness=0.42)
    m_screen = principled_mat(
        "MatScreen",
        base_color=(0.04, 0.08, 0.04, 1),
        emission=True,
        emission_color=(0.08, 0.98, 0.22),
        emission_strength=4.0,
    )
    m_cam = principled_mat("MatCam", base_color=(0.12, 0.13, 0.15, 1), metallic=0.45, roughness=0.48)
    m_card = principled_mat("MatCard", base_color=(0.55, 0.55, 0.55, 1), roughness=0.95)

    cyl_along_x("Cladding", R_CLAD, X_LEFT, X_RIGHT, mat=m_clad)
    cyl_along_x("Core", R_CORE, X_LEFT, X_RIGHT, mat=m_core)

    box_mesh(
        "PolishWindow",
        X_POL1 - X_POL0,
        POLISH_Y1 - POLISH_Y0,
        POLISH_HALF_Z * 2,
        ((X_POL0 + X_POL1) / 2, (POLISH_Y0 + POLISH_Y1) / 2, 0),
        mat=m_polish,
    )

    rng = random.Random(42)
    for i in range(55):
        bx = rng.uniform(X_POL0 + 0.05, X_POL1 - 0.05)
        by = rng.uniform(-R_CORE * 0.2, R_CORE * 0.65)
        bz = rng.uniform(-R_CLAD * 0.55, R_CLAD * 0.55)
        r = 0.035 + rng.random() * 0.042
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=r, location=(bx, by, bz))
        so = bpy.context.active_object
        so.name = f"Scat_{i}"
        em = principled_mat(
            f"ScatM_{i}",
            base_color=(0.95, 1.0, 0.55, 1),
            roughness=1.0,
            emission=True,
            emission_color=(0.9, 1.0, 0.45),
            emission_strength=1.5 + rng.random() * 1.2,
        )
        so.data.materials.append(em)

    cyl_along_x("RedBeam_in", 0.075, -6.4, X_LEFT - 0.02, mat=m_red)
    cyl_along_x("RedBeam_out", 0.058, X_LEFT, X_RIGHT + 0.85, mat=m_red)

    polish_hit = Vector(((X_POL0 + X_POL1) / 2, POLISH_Y1 + 0.06, 0))
    p_start = Vector((-2.4, 6.2, 0.35))
    d = (polish_hit - p_start).normalized()
    step = 1.05
    t0 = 0.35
    pts = [p_start + d * (t0 + i * step) for i in range(6)]
    p_be, p_cl, p_slm, p_fo, p_pre_hit = pts[0], pts[1], pts[2], pts[3], pts[4]

    for a, b, nm, rad in (
        (p_start, p_be, "GV_0", 0.055),
        (p_be, p_cl, "GV_1", 0.05),
        (p_cl, p_slm, "GV_2", 0.048),
        (p_slm, p_fo, "GV_3", 0.045),
        (p_fo, polish_hit, "GV_4", 0.038),
    ):
        cylinder_between(nm, a, b, rad, m_green)

    box_mesh("GreenLaser", 1.15, 0.78, 0.68, tuple(p_start + Vector((-0.5, 0.25, 0))), (0.18, -0.4, 0.06), m_metal)
    add_subsurf(bpy.data.objects["GreenLaser"], 2)

    box_mesh("RedLaser", 1.35, 0.82, 0.72, (-6.88, 0, 0), (0, 0, 0), m_metal)
    add_subsurf(bpy.data.objects["RedLaser"], 2)

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.4, depth=0.5, location=tuple((p_start + p_be) / 2), rotation=(0.65, 0.0, -0.55))
    bpy.context.active_object.name = "BeamExpander"
    bpy.context.active_object.data.materials.append(m_metal)

    axis_seg = lambda a, b: (Vector(b) - Vector(a)).normalized()
    lens_disk("LensBE", (Vector(p_be) + Vector(p_cl)) / 2, axis_seg(p_be, p_cl), 0.44, 0.14, m_glass)
    lens_disk("LensCol", (Vector(p_cl) + Vector(p_slm)) / 2, axis_seg(p_cl, p_slm), 0.42, 0.13, m_glass)
    lens_disk("LensFocus", (Vector(p_fo) + polish_hit) / 2, axis_seg(p_fo, polish_hit), 0.5, 0.2, m_glass)

    slm_axis = axis_seg(p_cl, p_slm)
    slm_mid = (Vector(p_cl) + Vector(p_slm)) / 2
    slm_quat = slm_axis.to_track_quat("X", "Y")
    e = slm_quat.to_euler()
    box_mesh("SLM_body", 0.92, 0.14, 0.7, tuple(slm_mid), (e.x, e.y, e.z), m_slm)
    off = Vector(slm_axis).normalized() * 0.09
    box_mesh("SLM_scr", 0.76, 0.05, 0.56, tuple(slm_mid + off), (e.x, e.y, e.z), m_screen)

    cam_x = X_RIGHT + 1.45
    box_mesh("CamBody", 1.05, 0.95, 0.78, (cam_x, 0, 0), (0, 0, 0), m_cam)
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.36, depth=0.9, location=(cam_x - 0.82, 0, 0), rotation=(0, math.pi / 2, 0))
    bpy.context.active_object.name = "CamLens"
    bpy.context.active_object.data.materials.append(m_glass)
    box_mesh("Speckle", 1.0, 0.04, 1.0, (cam_x + 1.12, 0, 0), (0, 0, 0), m_card)

    bpy.ops.object.light_add(type="AREA", location=(5.5, -10.5, 14.5))
    L1 = bpy.context.active_object
    L1.data.energy = 1200
    L1.data.size = 8
    bpy.ops.object.light_add(type="AREA", location=(-7, 6, 8))
    L2 = bpy.context.active_object
    L2.data.energy = 650
    L2.data.size = 5

    bpy.ops.object.camera_add(location=(9.2, -24.0, 12.5))
    cam_main = bpy.context.active_object
    cam_main.name = "CamMain"
    cam_main.rotation_euler = Euler((math.radians(56), 0, math.radians(20)), "XYZ")
    cam_main.data.type = "ORTHO"
    cam_main.data.ortho_scale = 17.0

    bpy.ops.object.camera_add(location=((X_POL0 + X_POL1) / 2 + 0.15, -5.2, 3.6))
    cam_close = bpy.context.active_object
    cam_close.name = "CamPolishClose"
    cam_close.rotation_euler = Euler((math.radians(74), 0, math.radians(8)), "XYZ")
    cam_close.data.type = "ORTHO"
    cam_close.data.ortho_scale = 4.0

    scene.camera = cam_main
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print("Saved:", BLEND_PATH)

    scene.render.filepath = str(MAIN_RENDER)
    bpy.ops.render.render(write_still=True)
    print("Rendered:", MAIN_RENDER)

    scene.camera = cam_close
    scene.render.filepath = str(CLOSEUP_RENDER)
    bpy.ops.render.render(write_still=True)
    print("Rendered:", CLOSEUP_RENDER)


if __name__ == "__main__":
    main()
