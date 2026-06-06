from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading

import pytest
from rich.console import Console

from paddle import __main__ as paddle_main
from paddle.monitor import (
    DiskMetric,
    GPUMetric,
    HostMetric,
    MonitorSnapshot,
    collect_host,
    collect_snapshot,
    main,
    parse_probe_output,
    read_nodelist,
    render_snapshot,
    snapshot_json,
)


PROBE_OUTPUT = """\
__CPU0__
cpu  100 0 50 850 0 0 0 0 0 0
__CPU1__
cpu  120 0 60 920 0 0 0 0 0 0
__CPU_PROCESSES__
alice 123 75.0 4.0 python train.py --epochs 10
root 1 0.1 0.2 /usr/lib/systemd/systemd --system
__PID_DETAILS__
123 alice python train.py --epochs 10
1 root /usr/lib/systemd/systemd --system
__DISKS__
Filesystem 1-blocks Used Available Capacity Mounted on
/dev/mapper/root 1000 600 400 60% /
server:/home 1000 900 100 90% /home/shared
tmpfs 100 1 99 1% /run
__GPUS__
0, GPU-abc, NVIDIA A100, 80, 1000, 40000, 70
__GPU_PROCESSES__
GPU-abc, 123, python, 1000
__END__
"""


def test_read_nodelist_ignores_comments_blanks_and_duplicates(tmp_path: Path) -> None:
    nodelist = tmp_path / "nodelist"
    nodelist.write_text("# cluster\n\ndart1\ndart2 # spare\ndart1\n", encoding="utf-8")
    assert read_nodelist(nodelist) == ["dart1", "dart2"]


def test_read_nodelist_rejects_missing_empty_and_multiword_entries(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot read nodelist"):
        read_nodelist(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.write_text("# none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no hosts"):
        read_nodelist(empty)

    invalid = tmp_path / "invalid"
    invalid.write_text("dart1 dart2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_nodelist(invalid)


def test_parse_probe_output_collects_cpu_gpu_processes_and_local_disks() -> None:
    metric = parse_probe_output("dart1", PROBE_OUTPUT, top=1)
    assert metric.status == "ok"
    assert metric.cpu_percent == 30.0
    assert [(process.user, process.pid) for process in metric.cpu_processes] == [
        ("alice", 123)
    ]
    assert metric.cpu_processes[0].command == "python train.py --epochs 10"
    assert len(metric.gpus) == 1
    assert metric.gpus[0].name == "NVIDIA A100"
    assert metric.gpus[0].processes[0].user == "alice"
    assert metric.gpus[0].processes[0].command == "python train.py --epochs 10"
    assert [(disk.source, disk.mount) for disk in metric.disks] == [
        ("/dev/mapper/root", "/")
    ]


def test_parse_probe_output_accepts_gpu_less_host() -> None:
    output = PROBE_OUTPUT.replace(
        "0, GPU-abc, NVIDIA A100, 80, 1000, 40000, 70", ""
    ).replace("GPU-abc, 123, python, 1000", "")
    metric = parse_probe_output("cpu-only", output, top=5)
    assert metric.gpus == []


def test_collect_host_reports_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 2)

    monkeypatch.setattr("paddle.monitor.subprocess.run", fake_run)
    metric = collect_host("offline", timeout=2, top=5)
    assert metric.status == "error"
    assert metric.error == "SSH probe timed out after 2s"


def test_collect_host_reports_ssh_failure(monkeypatch) -> None:
    calls: list[list[str]] = []
    completed = subprocess.CompletedProcess(
        ["ssh"], returncode=255, stdout="", stderr="banner\nconnection refused\n"
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        return completed

    monkeypatch.setattr("paddle.monitor.subprocess.run", fake_run)
    metric = collect_host("offline", timeout=2, top=5)
    assert metric.status == "error"
    assert metric.error == "connection refused"
    assert calls[0][calls[0].index("--") + 1] == "offline"


def test_collect_snapshot_preserves_nodelist_order_and_failures() -> None:
    def collector(host: str, timeout: float, top: int) -> HostMetric:
        if host == "bad":
            raise RuntimeError("failed")
        return HostMetric(host, "ok", cpu_percent=1.0)

    snapshot = collect_snapshot(["good", "bad"], 1, 2, collector=collector)
    assert [host.host for host in snapshot.hosts] == ["good", "bad"]
    assert snapshot.hosts[1].error == "failed"


def test_collect_snapshot_probes_hosts_concurrently() -> None:
    barrier = threading.Barrier(3, timeout=1)

    def collector(host: str, timeout: float, top: int) -> HostMetric:
        barrier.wait()
        return HostMetric(host, "ok", cpu_percent=1.0)

    snapshot = collect_snapshot(["dart1", "dart2", "dart3"], 1, 2, collector=collector)
    assert [host.host for host in snapshot.hosts] == ["dart1", "dart2", "dart3"]


def test_snapshot_json_and_render_include_metrics() -> None:
    snapshot = MonitorSnapshot(
        "2026-06-06T00:00:00+00:00",
        [
            HostMetric(
                "dart1",
                "ok",
                cpu_percent=12.5,
                gpus=[GPUMetric(0, "uuid", "NVIDIA A100", 50, 100, 1000, 60)],
                disks=[DiskMetric("/dev/root", "/", 50, 100, 50, 50)],
            )
        ],
    )
    assert json.loads(snapshot_json(snapshot))["hosts"][0]["cpu_percent"] == 12.5

    console = Console(record=True, width=160)
    console.print(render_snapshot(snapshot))
    rendered = console.export_text()
    assert "dart1" in rendered
    assert "12.5%" in rendered
    assert "0: 50%" in rendered
    assert "NVIDIA A100 | 0: 100/1000 MiB" in rendered
    assert "/dev/root" in rendered


def test_monitor_main_json_uses_nodelist(tmp_path: Path, monkeypatch, capsys) -> None:
    nodelist = tmp_path / "nodes"
    nodelist.write_text("dart1\n", encoding="utf-8")
    snapshot = MonitorSnapshot(
        "2026-06-06T00:00:00+00:00", [HostMetric("dart1", "ok", cpu_percent=1)]
    )
    monkeypatch.setattr("paddle.monitor.collect_snapshot", lambda *args: snapshot)
    assert main(["--nodelist", str(nodelist), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["hosts"][0]["host"] == "dart1"


def test_paddle_main_dispatches_monitor(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        paddle_main, "monitor_main", lambda args: calls.append(list(args)) or 7
    )
    assert paddle_main.main(["monitor", "--json"]) == 7
    assert calls == [["--json"]]


def test_paddle_monitor_help_is_forwarded(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        paddle_main.main(["monitor", "--help"])
    assert exc.value.code == 0
    assert "--nodelist" in capsys.readouterr().out
