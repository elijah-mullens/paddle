from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "example_py_scripts"


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _write_yaml(tmp_path: Path, source_name: str, updates: dict) -> Path:
    with open(EXAMPLE_DIR / source_name, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key, value in updates.items():
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            if isinstance(cursor, list):
                cursor = cursor[int(part)]
            else:
                cursor = cursor[part]
        leaf = parts[-1]
        if isinstance(cursor, list):
            cursor[int(leaf)] = value
        else:
            cursor[leaf] = value
    target = tmp_path / source_name
    with open(target, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return target


def _run_single(
    script_name: str, yaml_path: Path | None, extra_args: list[str] | None = None
) -> None:
    env = _base_env()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(_get_free_port())
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"

    cmd = [sys.executable, str(EXAMPLE_DIR / script_name)]
    if yaml_path is not None:
        cmd.extend(["--input", str(yaml_path)])
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(
        cmd,
        cwd=yaml_path.parent if yaml_path is not None else EXAMPLE_DIR,
        env=env,
        check=True,
        timeout=300,
    )


def _run_distributed(script_name: str, yaml_path: Path, nproc: int) -> None:
    env = _base_env()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(EXAMPLE_DIR / script_name),
        "--input",
        str(yaml_path),
    ]
    subprocess.run(
        cmd,
        cwd=yaml_path.parent,
        env=env,
        check=True,
        timeout=600,
    )


def test_shock_script(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "shock.yaml",
        {
            "geometry.cells.nx1": 48,
            "geometry.cells.nx2": 48,
            "integration.nlim": 2,
            "integration.tlim": 0.05,
            "outputs.0.dt": 0.05,
            "outputs.1.dt": 0.05,
        },
    )
    _run_single("shock.py", yaml_path)


def test_straka_script(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "straka.yaml",
        {
            "geometry.cells.nx1": 24,
            "geometry.cells.nx2": 48,
            "integration.nlim": 2,
            "integration.tlim": 5.0,
            "outputs.0.dt": 5.0,
            "outputs.1.dt": 5.0,
        },
    )
    _run_single("straka.py", yaml_path)


def test_robert_script(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "robert.yaml",
        {
            "geometry.cells.nx1": 48,
            "geometry.cells.nx2": 48,
            "distribute.nb2": 4,
            "distribute.nb3": 1,
            "integration.nlim": 2,
            "integration.tlim": 2.0,
            "outputs.0.dt": 5.0,
            "outputs.1.dt": 5.0,
        },
    )
    _run_distributed("robert.py", yaml_path, nproc=4)


def test_shallow_yz_script(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "shallow_yz.yaml",
        {
            "geometry.cells.nx2": 48,
            "geometry.cells.nx3": 48,
            "distribute.nb2": 2,
            "distribute.nb3": 2,
            "integration.nlim": 2,
            "integration.tlim": 0.2,
            "outputs.0.dt": 0.2,
        },
    )
    _run_distributed("shallow_yz.py", yaml_path, nproc=4)


def test_explosion_script(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "explosion.yaml",
        {
            "geometry.cells.nx1": 24,
            "geometry.cells.nx2": 24,
            "geometry.cells.nx3": 24,
            "integration.nlim": 2,
            "integration.tlim": 0.02,
            "outputs.0.dt": 0.02,
        },
    )
    _run_single("explosion.py", yaml_path)


def test_earth_moist_script(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "earth_moist.yaml",
        {
            "geometry.cells.nx1": 32,
            "integration.nlim": 1,
            "outputs.0.dt": 10.0,
        },
    )
    _run_single("earth_moist.py", yaml_path)


def test_saturn_adiabat_script(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "saturn1d.yaml",
        {
            "geometry.cells.nx1": 64,
        },
    )
    output = tmp_path / "saturn_profile.txt"
    _run_single(
        "test_saturn_adiabat.py",
        yaml_path,
        extra_args=["--output", str(output)],
    )
    assert output.exists()
