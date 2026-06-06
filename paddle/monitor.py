from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .nodes import DEFAULT_NODELIST, read_nodelist


REMOTE_PROBE = r"""
echo __CPU0__
head -n 1 /proc/stat
sleep 0.2
echo __CPU1__
head -n 1 /proc/stat
echo __CPU_PROCESSES__
LC_ALL=C ps -eo user=,pid=,pcpu=,pmem=,args= --sort=-pcpu
echo __PID_DETAILS__
LC_ALL=C ps -eo pid=,user=,args=
echo __DISKS__
LC_ALL=C df -P -B1 -l
echo __GPUS__
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
fi
echo __GPU_PROCESSES__
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
fi
echo __END__
"""


@dataclass
class ProcessMetric:
    user: str
    pid: int
    cpu_percent: float | None
    memory_percent: float | None
    command: str


@dataclass
class GPUProcessMetric:
    gpu_uuid: str
    pid: int
    user: str
    command: str
    memory_mib: int


@dataclass
class GPUMetric:
    index: int
    uuid: str
    name: str
    utilization_percent: int
    memory_used_mib: int
    memory_total_mib: int
    temperature_c: int
    processes: list[GPUProcessMetric] = field(default_factory=list)


@dataclass
class DiskMetric:
    source: str
    mount: str
    used_bytes: int
    total_bytes: int
    available_bytes: int
    usage_percent: int


@dataclass
class HostMetric:
    host: str
    status: str
    error: str | None = None
    cpu_percent: float | None = None
    cpu_processes: list[ProcessMetric] = field(default_factory=list)
    gpus: list[GPUMetric] = field(default_factory=list)
    disks: list[DiskMetric] = field(default_factory=list)


@dataclass
class MonitorSnapshot:
    collected_at: str
    hosts: list[HostMetric]


def _number(value: str, converter: Callable[[float], int | float]) -> int | float:
    value = value.strip()
    if value in {"", "N/A", "[Not Supported]"}:
        return converter(0)
    return converter(float(value))


def _cpu_percent(first: str, second: str) -> float:
    def counters(line: str) -> tuple[int, int]:
        values = [int(item) for item in line.split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    total_first, idle_first = counters(first)
    total_second, idle_second = counters(second)
    total_delta = total_second - total_first
    if total_delta <= 0:
        return 0.0
    return round(100.0 * (1.0 - (idle_second - idle_first) / total_delta), 1)


def _sections(output: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in output.splitlines():
        if line.startswith("__") and line.endswith("__"):
            name = line.strip("_")
            current = sections.setdefault(name, [])
        elif current is not None:
            current.append(line)
    return sections


def parse_probe_output(host: str, output: str, top: int) -> HostMetric:
    sections = _sections(output)
    required = {"CPU0", "CPU1", "CPU_PROCESSES", "PID_DETAILS", "DISKS", "GPUS"}
    missing = sorted(required - sections.keys())
    if missing:
        raise ValueError(f"incomplete probe output; missing {', '.join(missing)}")
    if not sections["CPU0"] or not sections["CPU1"]:
        raise ValueError("incomplete CPU counters")

    processes: list[ProcessMetric] = []
    for line in sections["CPU_PROCESSES"][:top]:
        fields = line.split(None, 4)
        if len(fields) != 5:
            continue
        user, pid, cpu, memory, command = fields
        processes.append(
            ProcessMetric(user, int(pid), float(cpu), float(memory), command)
        )

    pid_details: dict[int, tuple[str, str]] = {}
    for line in sections["PID_DETAILS"]:
        fields = line.split(None, 2)
        if len(fields) >= 2:
            pid_details[int(fields[0])] = (
                fields[1],
                fields[2] if len(fields) == 3 else "",
            )

    gpu_processes: list[GPUProcessMetric] = []
    for row in csv.reader(sections.get("GPU_PROCESSES", []), skipinitialspace=True):
        if len(row) != 4:
            continue
        uuid, pid, process_name, memory = row
        process_pid = int(pid)
        user, full_command = pid_details.get(process_pid, ("?", process_name.strip()))
        gpu_processes.append(
            GPUProcessMetric(
                uuid.strip(),
                process_pid,
                user,
                full_command,
                int(_number(memory, int)),
            )
        )

    gpus: list[GPUMetric] = []
    for row in csv.reader(sections["GPUS"], skipinitialspace=True):
        if len(row) != 7:
            continue
        index, uuid, name, utilization, memory_used, memory_total, temperature = row
        gpu = GPUMetric(
            int(index),
            uuid.strip(),
            name.strip(),
            int(_number(utilization, int)),
            int(_number(memory_used, int)),
            int(_number(memory_total, int)),
            int(_number(temperature, int)),
        )
        gpu.processes = [
            process for process in gpu_processes if process.gpu_uuid == gpu.uuid
        ][:top]
        gpus.append(gpu)

    disks: list[DiskMetric] = []
    for line in sections["DISKS"][1:]:
        fields = line.split()
        if len(fields) < 6 or not fields[0].startswith("/dev/"):
            continue
        source, total, used, available, percent = fields[:5]
        mount = " ".join(fields[5:])
        disks.append(
            DiskMetric(
                source,
                mount,
                int(used),
                int(total),
                int(available),
                int(percent.rstrip("%")),
            )
        )

    return HostMetric(
        host=host,
        status="ok",
        cpu_percent=_cpu_percent(sections["CPU0"][0], sections["CPU1"][0]),
        cpu_processes=processes,
        gpus=gpus,
        disks=disks,
    )


def collect_host(host: str, timeout: float, top: int) -> HostMetric:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        "--",
        host,
        "sh",
        "-s",
    ]
    try:
        result = subprocess.run(
            command,
            input=REMOTE_PROBE,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return HostMetric(host, "error", f"SSH probe timed out after {timeout:g}s")
    except OSError as exc:
        return HostMetric(host, "error", str(exc))

    if result.returncode != 0:
        error = result.stderr.strip().splitlines()
        return HostMetric(
            host, "error", error[-1] if error else f"SSH exited {result.returncode}"
        )
    try:
        return parse_probe_output(host, result.stdout, top)
    except (ValueError, IndexError) as exc:
        return HostMetric(host, "error", str(exc))


def collect_snapshot(
    hosts: Sequence[str],
    timeout: float,
    top: int,
    collector: Callable[[str, float, int], HostMetric] = collect_host,
) -> MonitorSnapshot:
    metrics: dict[str, HostMetric] = {}
    with ThreadPoolExecutor(max_workers=min(32, len(hosts))) as executor:
        futures = {
            executor.submit(collector, host, timeout, top): host for host in hosts
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                metrics[host] = future.result()
            except Exception as exc:
                metrics[host] = HostMetric(host, "error", str(exc))
    return MonitorSnapshot(
        datetime.now(timezone.utc).isoformat(), [metrics[host] for host in hosts]
    )


def snapshot_json(snapshot: MonitorSnapshot) -> str:
    return json.dumps(asdict(snapshot), indent=2)


def _bytes(value: int) -> str:
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or suffix == "TiB":
            return f"{size:.1f}{suffix}"
        size /= 1024
    return f"{size:.1f}TiB"


def render_snapshot(snapshot: MonitorSnapshot) -> Table:
    outer = Table.grid(expand=True)
    outer.add_row(f"[bold]Collected:[/bold] {snapshot.collected_at}")

    summary = Table(title="Hosts", expand=True)
    for heading in ("Host", "CPU", "GPU", "GPU memory", "Fullest disk", "Status"):
        summary.add_column(heading)
    for host in snapshot.hosts:
        if host.status != "ok":
            summary.add_row(host.host, "-", "-", "-", "-", f"[red]{host.error}[/red]")
            continue
        fullest = max(host.disks, key=lambda disk: disk.usage_percent, default=None)
        gpu_util = ", ".join(
            f"{gpu.index}: {gpu.utilization_percent}%" for gpu in host.gpus
        )
        gpu_memory_values = ", ".join(
            f"{gpu.index}: {gpu.memory_used_mib}/{gpu.memory_total_mib} MiB"
            for gpu in host.gpus
        )
        gpu_zero = next((gpu for gpu in host.gpus if gpu.index == 0), None)
        gpu_memory = (
            f"{gpu_zero.name} | {gpu_memory_values}" if gpu_zero else gpu_memory_values
        )
        disk = f"{fullest.mount} {fullest.usage_percent}%" if fullest else "-"
        summary.add_row(
            host.host,
            f"{host.cpu_percent:.1f}%",
            gpu_util or "-",
            gpu_memory or "-",
            disk,
            "[green]ok[/green]",
        )
    outer.add_row(summary)

    for host in snapshot.hosts:
        if host.status != "ok":
            continue
        details = Table(title=host.host, expand=True)
        details.add_column("Resource")
        details.add_column("ID / Mount")
        details.add_column("User")
        details.add_column("Usage")
        details.add_column("Command / Device")
        for process in host.cpu_processes:
            details.add_row(
                "CPU process",
                str(process.pid),
                process.user,
                f"{process.cpu_percent:.1f}% CPU, {process.memory_percent:.1f}% MEM",
                process.command,
            )
        for gpu in host.gpus:
            details.add_row(
                "GPU",
                str(gpu.index),
                "",
                (
                    f"{gpu.utilization_percent}%, "
                    f"{gpu.memory_used_mib}/{gpu.memory_total_mib} MiB, "
                    f"{gpu.temperature_c} C"
                ),
                gpu.name,
            )
            for process in gpu.processes:
                details.add_row(
                    "GPU process",
                    str(process.pid),
                    process.user,
                    f"{process.memory_mib} MiB",
                    process.command,
                )
        for disk in host.disks:
            details.add_row(
                "Disk",
                disk.mount,
                "",
                f"{disk.usage_percent}% ({_bytes(disk.used_bytes)}/{_bytes(disk.total_bytes)})",
                disk.source,
            )
        outer.add_row(details)
    return outer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle monitor",
        description="Monitor CPU, GPU, and disk usage on machines over SSH.",
    )
    parser.add_argument(
        "--nodelist",
        type=Path,
        default=DEFAULT_NODELIST,
        help=f"One-SSH-alias-per-line file (default: {DEFAULT_NODELIST})",
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        type=float,
        const=5.0,
        metavar="SECONDS",
        help="Refresh continuously, every 5 seconds by default.",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON snapshot.")
    parser.add_argument("--top", type=int, default=5, help="Top processes per host.")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="SSH probe timeout in seconds."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error("--top must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.watch is not None and args.watch <= 0:
        parser.error("--watch interval must be positive")
    if args.json and args.watch is not None:
        parser.error("--json and --watch cannot be used together")
    try:
        hosts = read_nodelist(args.nodelist)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        snapshot = collect_snapshot(hosts, args.timeout, args.top)
        print(snapshot_json(snapshot))
        return int(any(host.status != "ok" for host in snapshot.hosts))

    console = Console()
    if args.watch is None:
        snapshot = collect_snapshot(hosts, args.timeout, args.top)
        console.print(render_snapshot(snapshot))
        return int(any(host.status != "ok" for host in snapshot.hosts))

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                live.update(
                    render_snapshot(collect_snapshot(hosts, args.timeout, args.top)),
                    refresh=True,
                )
                time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
