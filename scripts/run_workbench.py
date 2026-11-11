"""Single-command launcher for the VLM-Steer Web Workbench.

Starts both:
1. Python Starlette/Uvicorn inference API server (http://127.0.0.1:8000)
2. Next.js interactive web frontend (http://localhost:3000)

Usage:
    python scripts/run_workbench.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Terminate a child process and its descendant tree cross-platform."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    npm_cmd = shutil.which("npm") or "npm"

    print("==========================================================================")
    print("  [VLM-Steer] Launching Next.js Workbench + Python Inference API")
    print("==========================================================================")
    print("  * Python API  -> http://127.0.0.1:8000/api/health")
    print("  * Next.js UI  -> http://localhost:3000")
    print("  Press Ctrl+C to shut down both servers.\n")

    api_proc = subprocess.Popen(
        [
            python_exe,
            "-m",
            "uvicorn",
            "api.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(REPO_ROOT),
    )

    web_proc = subprocess.Popen(
        [npm_cmd, "--prefix", "web", "run", "dev"],
        cwd=str(REPO_ROOT),
        shell=(os.name == "nt"),
    )

    try:
        while True:
            if api_proc.poll() is not None:
                print("\n[Error] Python API server exited unexpectedly.")
                break
            if web_proc.poll() is not None:
                print("\n[Error] Next.js dev server exited unexpectedly.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping VLM-Steer servers...")
    finally:
        _terminate_tree(web_proc)
        _terminate_tree(api_proc)
        print("[OK] Clean shutdown complete.")


if __name__ == "__main__":
    main()
