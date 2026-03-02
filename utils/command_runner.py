import subprocess
import threading
import time
import sys
from typing import Callable, List


def _resolve_cmd(cmd: List[str]) -> List[str]:
    if sys.platform == "win32" and str(cmd[0]).lower().endswith(".bat"):
        return ["cmd.exe", "/c"] + [str(c) for c in cmd]
    return [str(c) for c in cmd]


def run_command(
    cmd: List[str],
    log_callback: Callable[[str], None] = None,
    cwd: str = None,
    abort_event: threading.Event = None,
) -> int:
    """
    Run a subprocess, streaming both stdout AND stderr concurrently on
    separate threads to prevent pipe buffer deadlocks on Windows.
    Both streams are logged live. Full output included in error messages.
    """
    cmd = _resolve_cmd(cmd)

    if log_callback:
        log_callback(f"$ {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        encoding="utf-8",
        errors="replace",
    )

    stdout_lines = []
    stderr_lines = []
    last_output_time = [time.time()]
    stop_event = threading.Event()

    # ── Stderr reader (concurrent) ────────────────────────────────────────────
    def read_stderr():
        for line in process.stderr:
            line = line.rstrip()
            if line:
                stderr_lines.append(line)
                last_output_time[0] = time.time()
                if log_callback:
                    log_callback(f"[stderr] {line}")

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    # ── Abort watcher ─────────────────────────────────────────────────────────
    def abort_watcher():
        if abort_event is None:
            return
        abort_event.wait()
        if process.poll() is None:
            if log_callback:
                log_callback("  ⚠ Abort requested — killing process...")
            process.kill()
        stop_event.set()

    abort_thread = threading.Thread(target=abort_watcher, daemon=True)
    abort_thread.start()

    # ── Heartbeat ─────────────────────────────────────────────────────────────
    def heartbeat():
        while not stop_event.wait(15):
            if stop_event.is_set():
                break
            silent_for = int(time.time() - last_output_time[0])
            if silent_for >= 14 and log_callback:
                log_callback(f"  ... still running (silent for {silent_for}s, process is alive)")

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    # ── Stream stdout live (main thread) ──────────────────────────────────────
    for line in process.stdout:
        line = line.rstrip()
        if line:
            stdout_lines.append(line)
            last_output_time[0] = time.time()
            if log_callback:
                log_callback(line)

    process.wait()
    stderr_thread.join(timeout=5)
    stop_event.set()

    if abort_event and abort_event.is_set():
        raise RuntimeError("ABORTED")

    if process.returncode != 0:
        # Include both stdout and stderr in error — different tools use different streams
        all_output = []
        if stdout_lines:
            all_output.append("STDOUT:\n" + "\n".join(stdout_lines[-20:]))
        if stderr_lines:
            all_output.append("STDERR:\n" + "\n".join(stderr_lines[-20:]))
        output_dump = "\n".join(all_output) if all_output else "(no output captured)"
        raise RuntimeError(
            f"Command failed (exit code {process.returncode}):\n"
            f"  CMD: {' '.join(cmd)}\n"
            f"  {output_dump}"
        )

    return process.returncode
