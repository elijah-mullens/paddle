from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "example_py_scripts"
RUN_HYDRO = EXAMPLE_DIR / "run_hydro.py"
GCM_NPROC = 2


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize(
    "yaml_name",
    [
        "jupiter_crm_small.yaml",
        "jupiter_crm_dry_small.yaml",
        "jupiter_gcm_small.yaml",
        "jupiter_gcm_dry_small.yaml",
    ],
)
def test_run_hydro_jupiter_cpu_examples(tmp_path: Path, yaml_name: str) -> None:
    input_path = tmp_path / yaml_name
    shutil.copy2(EXAMPLE_DIR / yaml_name, input_path)

    env = os.environ.copy()
    python_path = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["BACKEND"] = "gloo"
    env["DEVICE"] = "cpu"
    env.pop("DEVICE_ID", None)
    if "gcm" in yaml_name:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={GCM_NPROC}",
            str(RUN_HYDRO),
            "-i",
            str(input_path),
        ]
        timeout = 600
    else:
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(_get_free_port())
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["LOCAL_RANK"] = "0"
        cmd = [sys.executable, str(RUN_HYDRO), "-i", str(input_path)]
        timeout = 300

    subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=timeout,
    )
