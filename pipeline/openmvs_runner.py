import threading
from pathlib import Path
from typing import Callable
from utils.command_runner import run_command
from core.exceptions import OpenMVSError


def _w(p) -> str:
    """Resolve to absolute path."""
    return str(Path(p).resolve())


class OpenMVSRunner:
    def __init__(self, exes: dict, use_gpu: bool = True):
        self.interface_colmap = exes["interface_colmap"]
        self.densify = exes["densify"]
        self._reconstruct_mesh_exe = exes["reconstruct_mesh"]
        self._refine_mesh_exe = exes["refine_mesh"]
        self._texture_mesh_exe = exes["texture_mesh"]
        self.use_gpu = use_gpu

    def _run(self, exe: str, args: list, cwd: str = None, log_callback=None, abort_event=None):
        cmd = [exe] + args
        try:
            run_command(cmd, log_callback=log_callback, cwd=cwd, abort_event=abort_event)
        except RuntimeError as e:
            raise OpenMVSError(str(e))

    def convert_to_mvs(self, sparse_model_path, output_mvs, image_folder, working_folder, log_callback=None, abort_event=None):
        Path(output_mvs).parent.mkdir(parents=True, exist_ok=True)
        self._run(self.interface_colmap, [
            "-i", _w(sparse_model_path),
            "-o", _w(output_mvs),
            "--image-folder", _w(image_folder),
            "-w", _w(working_folder),
        ], cwd=_w(working_folder), log_callback=log_callback, abort_event=abort_event)

    def densify_point_cloud(self, scene_mvs, working_folder, log_callback=None, abort_event=None):
        args = [_w(scene_mvs)]
        if not self.use_gpu:
            args += ["--cuda-device", "-1"]
        self._run(self.densify, args,
                  cwd=_w(working_folder), log_callback=log_callback, abort_event=abort_event)

    def reconstruct_mesh(self, scene_dense_mvs, working_folder, log_callback=None, abort_event=None):
        self._run(self._reconstruct_mesh_exe, [_w(scene_dense_mvs)],
                  cwd=_w(working_folder), log_callback=log_callback, abort_event=abort_event)

    def refine_mesh(self, scene_mesh_mvs, working_folder, log_callback=None, abort_event=None):
        args = [_w(scene_mesh_mvs)]
        if not self.use_gpu:
            args += ["--cuda-device", "-1"]
        self._run(self._refine_mesh_exe, args,
                  cwd=_w(working_folder), log_callback=log_callback, abort_event=abort_event)

    def texture_mesh(self, scene_refine_mvs, working_folder, log_callback=None, abort_event=None):
        # --export-type obj forces output as .obj + .mtl + .png instead of .ply + .png
        self._run(self._texture_mesh_exe, [
            _w(scene_refine_mvs),
            "--export-type", "obj",
        ], cwd=_w(working_folder), log_callback=log_callback, abort_event=abort_event)
