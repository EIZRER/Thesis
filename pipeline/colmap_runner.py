import threading
from pathlib import Path
from typing import Callable
from utils.command_runner import run_command
from core.exceptions import COLMAPError


class COLMAPRunner:
    def __init__(self, colmap_exe: str, use_gpu: bool = True):
        bat_path = Path(colmap_exe)
        exe_path = bat_path.parent / "bin" / "colmap.exe"
        if bat_path.suffix.lower() == ".bat" and exe_path.exists():
            self.colmap_exe = str(exe_path)
        else:
            self.colmap_exe = str(bat_path)
        self.use_gpu = use_gpu

    def _run(self, args: list, log_callback: Callable = None, abort_event: threading.Event = None):
        cmd = [self.colmap_exe] + args
        try:
            run_command(cmd, log_callback=log_callback, abort_event=abort_event)
        except RuntimeError as e:
            raise COLMAPError(str(e))

    def create_database(self, database_path: str, log_callback=None, abort_event=None):
        self._run(["database_creator", "--database_path", str(database_path)], log_callback, abort_event)

    def _cpu_extract_flags(self) -> list:
        """Returns flags to force CPU for feature extraction (only when GPU is disabled).
        Setting gpu_index to -1 is the portable way to disable CUDA across COLMAP versions."""
        return ["--SiftExtraction.gpu_index", "-1"] if not self.use_gpu else []

    def _cpu_match_flags(self) -> list:
        """Returns flags to force CPU for feature matching (only when GPU is disabled)."""
        return ["--SiftMatching.gpu_index", "-1"] if not self.use_gpu else []

    def extract_features(self, database_path: str, image_path: str, log_callback=None, abort_event=None):
        self._run([
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.camera_model", "PINHOLE",
            *self._cpu_extract_flags(),
        ], log_callback, abort_event)

    def match_features_exhaustive(self, database_path: str, log_callback=None, abort_event=None):
        self._run([
            "exhaustive_matcher",
            "--database_path", str(database_path),
            *self._cpu_match_flags(),
        ], log_callback, abort_event)

    # Keep the old name as an alias so nothing else breaks
    match_features = match_features_exhaustive

    def match_features_sequential(self, database_path: str, overlap: int = 10, log_callback=None, abort_event=None):
        self._run([
            "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", str(overlap),
            *self._cpu_match_flags(),
        ], log_callback, abort_event)

    def match_features_vocab_tree(self, database_path: str, vocab_tree_path: str, log_callback=None, abort_event=None):
        if not vocab_tree_path or not Path(vocab_tree_path).is_file():
            raise COLMAPError(
                f"Vocab tree file not found: '{vocab_tree_path}'\n"
                "Download one from https://demuc.de/colmap/#download and set the path in Settings."
            )
        self._run([
            "vocab_tree_matcher",
            "--database_path", str(database_path),
            "--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path),
            *self._cpu_match_flags(),
        ], log_callback, abort_event)

    def run_mapper(self, database_path: str, image_path: str, output_path: str, log_callback=None, abort_event=None):
        self._run([
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
        ], log_callback, abort_event)

    def convert_model_to_txt(self, input_path: str, output_path: str, log_callback=None, abort_event=None):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self._run([
            "model_converter",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", "TXT",
        ], log_callback, abort_event)
