import shutil
from pathlib import Path
from core.logger import get_logger

logger = get_logger("model_converter")


def collect_outputs(mvs_dir: str, output_dir: str):
    """
    Copy final outputs from the mvs folder into the output folder.
    Handles both .obj (textured) and .ply (untextured) final outputs.
    """
    mvs = Path(mvs_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    copied = []

    def copy(src: Path, dest_name: str):
        dst = out / dest_name
        shutil.copy2(src, dst)
        copied.append(dst.name)
        logger.info(f"Copied {src.name} → output/{dst.name}")

    # ── Textured mesh: prefer .obj, fall back to textured .ply ───────────────
    obj_files = list(mvs.glob("*texture*.obj"))
    textured_ply = list(mvs.glob("*texture*.ply"))

    if obj_files:
        obj = obj_files[0]
        copy(obj, "textured_mesh.obj")
        # Copy the .mtl with same stem
        mtl = obj.with_suffix(".mtl")
        if mtl.exists():
            copy(mtl, "textured_mesh.mtl")
        # Copy all associated texture images
        for png in mvs.glob(obj.stem + "*.png"):
            copy(png, png.name)
        for jpg in mvs.glob(obj.stem + "*.jpg"):
            copy(jpg, jpg.name)
    elif textured_ply:
        copy(textured_ply[0], "textured_mesh.ply")
        # Copy the texture PNG alongside it
        for png in mvs.glob("*texture*.png"):
            copy(png, "textured_mesh.png")

    # ── Dense point cloud ─────────────────────────────────────────────────────
    dense_ply = mvs / "scene_dense.ply"
    if dense_ply.exists():
        copy(dense_ply, "dense_point_cloud.ply")

    # ── Raw mesh (before texture) ─────────────────────────────────────────────
    mesh_ply = mvs / "scene_dense_mesh.ply"
    if mesh_ply.exists():
        copy(mesh_ply, "mesh.ply")

    if not copied:
        logger.warning("No output files found to copy. Check mvs/ directory.")

    return copied
