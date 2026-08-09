#!/usr/bin/env python3
"""Ship the working tree to a Colab GPU VM and run a command there.

The Mac this is developed on has no NVIDIA GPU, so every build/test/benchmark
runs remotely. This packs the repo into a self-extracting payload and hands it
to `colab exec`, which is a single round trip.

    tools/colab_run.py -s dev "bash scripts/build.sh && python -m pytest -q"
    tools/colab_run.py -s dev --ensure --gpu A100 "nvidia-smi"

Kernel state persists between `colab exec` calls, but we re-extract every time
so the VM always matches local disk.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLAB = os.path.expanduser("~/.local/bin/colab")
REMOTE = "/content/nibblegemm"

SKIP_DIRS = {
    ".git", "__pycache__", "build", "dist", ".pytest_cache",
    ".ipynb_checkpoints", ".venv", "venv", ".mypy_cache", "artifacts",
}
SKIP_SUFFIX = (".so", ".o", ".pyc", ".tar.gz", ".ncu-rep", ".nsys-rep")


def build_payload_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if name.endswith(SKIP_SUFFIX):
                    continue
                full = os.path.join(dirpath, name)
                if os.path.getsize(full) > 4 << 20:
                    continue
                tf.add(full, arcname=os.path.relpath(full, ROOT))
    return buf.getvalue()


PAYLOAD = '''
import base64, io, os, subprocess, sys, tarfile

DEST = {remote!r}
os.makedirs(DEST, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BLOB)), mode="r:gz") as tf:
    tf.extractall(DEST)
os.chdir(DEST)

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = DEST + "/python:" + env.get("PYTHONPATH", "")

proc = subprocess.Popen(
    ["bash", "-lc", {cmd!r}],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=env, cwd=DEST,
)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print(f"\\n[remote] exit={{rc}}", flush=True)
if rc != 0:
    raise SystemExit(rc)
'''


def session_exists(session: str) -> bool:
    out = subprocess.run([COLAB, "sessions"], capture_output=True, text=True).stdout
    return session in out


def ensure_session(session: str, gpu: str) -> None:
    if session_exists(session):
        return
    print(f"[local] creating session {session!r} with --gpu {gpu}", file=sys.stderr)
    cmd = [COLAB, "new", "-s", session]
    if gpu.lower() != "cpu":
        cmd += ["--gpu", gpu]
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"[local] could not allocate {gpu}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", help="shell command to run inside the repo on the VM")
    ap.add_argument("-s", "--session", default="nibble")
    ap.add_argument("--ensure", action="store_true", help="create the session if absent")
    ap.add_argument("--gpu", default="A100")
    ap.add_argument(
        "--timeout",
        type=float,
        default=2400.0,
        help="seconds to wait for the remote call; the colab default of 30s "
        "does not survive an nvcc build",
    )
    args = ap.parse_args()

    if args.ensure:
        ensure_session(args.session, args.gpu)

    blob = base64.b64encode(build_payload_tar()).decode()
    print(f"[local] payload {len(blob)/1024:.0f} KiB -> {args.session}", file=sys.stderr)

    script = f'BLOB = "{blob}"\n' + PAYLOAD.format(remote=REMOTE, cmd=args.command)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        return subprocess.run(
            [COLAB, "exec", "-s", args.session, "-f", path, "--timeout", str(args.timeout)]
        ).returncode
    finally:
        os.unlink(path)


if __name__ == "__main__":
    raise SystemExit(main())
