import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def get_workspace_paths(workspace_dir: str) -> dict:
    ws = Path(workspace_dir)
    return {
        "root": ws,
        "database": ws / "database.db",
        "sparse": ws / "sparse",
        "sparse_model": ws / "sparse" / "0",           # binary .bin output from mapper
        "sparse_txt": ws / "sparse" / "0" / "sparse",  # TXT output from model_converter
        "mvs": ws / "mvs",
        "scene": ws / "mvs" / "scene.mvs",
        "scene_dense": ws / "mvs" / "scene_dense.mvs",
        "scene_mesh": ws / "mvs" / "scene_dense_mesh.mvs",
        "scene_refine": ws / "mvs" / "scene_dense_mesh_refine.mvs",
        "scene_texture": ws / "mvs" / "scene_dense_mesh_refine_texture.mvs",
        "output": ws / "output",
    }


def ensure_workspace(workspace_dir: str):
    paths = get_workspace_paths(workspace_dir)
    for key in ["root", "sparse", "sparse_model", "sparse_txt", "mvs", "output"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths
