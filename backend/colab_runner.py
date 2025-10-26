"""Utility helpers to launch the FastAPI backend from Google Colab.

This module spins up the Uvicorn server and (optionally) exposes it publicly
via LocalTunnel so the React frontend or external tools can reach the API.

The script is intentionally dependency-light so it can run inside a Colab
notebook cell after installing the Python requirements and the Node.js
`localtunnel` CLI.  Example usage inside Colab:

```python
!pip install -r backend/requirements.txt
!npm install -g localtunnel
!python backend/colab_runner.py --port 8000
```

The command prints both the local Uvicorn address and the public LocalTunnel
URL.  Stop the cell (Ctrl+C / Interrupt execution) to terminate the processes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from typing import Iterable, List, Optional, Tuple


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the FastAPI backend and optionally publish it through "
            "LocalTunnel.  Designed for execution inside Google Colab."
        )
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for Uvicorn (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port where Uvicorn and LocalTunnel should listen (default: 8000).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Log level forwarded to Uvicorn (default: info).",
    )
    parser.add_argument(
        "--no-localtunnel",
        action="store_true",
        help="Start only Uvicorn without exposing a public tunnel.",
    )
    parser.add_argument(
        "--subdomain",
        default=None,
        help="Optional LocalTunnel subdomain (requires a paid LocalTunnel plan).",
    )
    return parser.parse_args(argv)


def _stream_output(process: subprocess.Popen[str], prefix: str) -> threading.Thread:
    """Continuously stream process output to STDOUT in a background thread."""

    def _target() -> None:
        if process.stdout is None:
            return
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                print(f"{prefix} {line}", end="")
        except ValueError:
            # Stream may already be closed when the process terminates.
            pass

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread


def _start_uvicorn(host: str, port: int, log_level: str) -> subprocess.Popen[bytes]:
    command: List[str] = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]
    print(f"[uvicorn] Launching backend at http://{host}:{port} ...")
    process = subprocess.Popen(command)
    return process


def _localtunnel_command(port: int, subdomain: Optional[str]) -> List[str]:
    candidates: Tuple[Tuple[str, List[str]], ...] = (
        ("lt", ["lt"]),
        ("localtunnel", ["localtunnel"]),
        ("npx", ["npx", "localtunnel"]),
    )
    for executable, base in candidates:
        if shutil.which(executable):
            command = [*base, "--port", str(port)]
            if subdomain:
                command.extend(["--subdomain", subdomain])
            return command
    raise RuntimeError(
        "LocalTunnel CLI not found. Install it with `!npm install -g localtunnel`"
        " before running this launcher."
    )


def _start_localtunnel(port: int, subdomain: Optional[str]) -> Tuple[subprocess.Popen[str], Optional[str]]:
    command = _localtunnel_command(port, subdomain)
    print("[localtunnel] Starting tunnel ...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    public_url: Optional[str] = None
    if process.stdout is not None:
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                print(f"[localtunnel] {line}", end="")
                normalized = line.strip().lower()
                if normalized.startswith("your url is:"):
                    public_url = line.split(":", 1)[1].strip()
                    break
        except ValueError:
            pass
    _stream_output(process, "[localtunnel]")
    return process, public_url


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    uvicorn_process = _start_uvicorn(args.host, args.port, args.log_level)
    localtunnel_process: Optional[subprocess.Popen[str]] = None
    public_url: Optional[str] = None

    if not args.no_localtunnel:
        try:
            localtunnel_process, public_url = _start_localtunnel(args.port, args.subdomain)
        except RuntimeError as exc:
            print(f"[localtunnel] {exc}", file=sys.stderr)

    if public_url:
        print(f"[localtunnel] Public URL: {public_url}")
    elif not args.no_localtunnel:
        print("[localtunnel] Tunnel started, but the public URL was not detected.")

    print("\nPress Ctrl+C (Interrupt execution) to stop the server.")

    try:
        while True:
            time.sleep(1)
            if uvicorn_process.poll() is not None:
                print("[uvicorn] Process exited.")
                break
    except KeyboardInterrupt:
        print("\nShutting down services ...")
    finally:
        for process in (localtunnel_process, uvicorn_process):
            if process is None:
                continue
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
