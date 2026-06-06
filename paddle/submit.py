from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence
from uuid import uuid4

from .nodes import DEFAULT_NODELIST, read_nodelist


REMOTE_RUN = r"""
cwd=$1
shift

shopt -s expand_aliases
if ! source "${HOME}/.bash_profile"; then
  echo "cannot source remote bash profile: ${HOME}/.bash_profile" >&2
  exit 2
fi
set -eu

case "$cwd" in
  "~") cwd=$HOME ;;
  "~/"*) cwd=$HOME/${cwd#~/} ;;
esac

if ! cd -- "$cwd"; then
  echo "cannot use remote working directory: $cwd" >&2
  exit 3
fi

printf -v command_line '%q ' "$@"
eval "$command_line"
"""


@dataclass
class Submission:
    job_id: str
    host: str
    pid: int
    cwd: str
    log: str
    command: list[str]


def _remote_invocation(cwd: str, command: Sequence[str]) -> str:
    arguments = ["bash", cwd, *command]
    return "bash -c " + " ".join(
        shlex.quote(argument) for argument in [REMOTE_RUN, *arguments]
    )


def _last_error(log: Path, fallback: str) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        lines = []
    return lines[-1] if lines else fallback


def _job_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _local_log_path(log: str, job_id: str) -> Path:
    path = Path(log).expanduser() if log else Path.cwd() / f"{job_id}.log"
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def submit_job(
    host: str,
    command: Sequence[str],
    *,
    cwd: str = "~",
    log: str = "",
    timeout: float = 10.0,
) -> Submission:
    job_id = _job_id()
    local_log = _local_log_path(log, job_id)
    try:
        local_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = local_log.open("w", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot create local log {local_log}: {exc}") from exc

    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        "--",
        host,
        _remote_invocation(cwd, command),
    ]
    try:
        process = subprocess.Popen(
            ssh_command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_handle.close()
        raise ValueError(str(exc)) from exc
    finally:
        log_handle.close()

    try:
        returncode = process.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        returncode = None
    if returncode not in {None, 0}:
        raise ValueError(_last_error(local_log, f"SSH exited {returncode}"))

    return Submission(job_id, host, process.pid, cwd, str(local_log), list(command))


def submission_json(submission: Submission) -> str:
    return json.dumps(asdict(submission), indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle submit",
        description="Submit a detached command to a machine in the nodelist.",
    )
    parser.add_argument("host", help="SSH host or alias from the nodelist.")
    parser.add_argument(
        "--nodelist",
        type=Path,
        default=DEFAULT_NODELIST,
        help=f"One-SSH-alias-per-line file (default: {DEFAULT_NODELIST})",
    )
    parser.add_argument(
        "--cwd",
        default="~",
        help="Remote working directory (default: remote home directory).",
    )
    parser.add_argument(
        "--log",
        default="",
        help="Local log path (default: generated in the current local directory).",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="SSH connection timeout in seconds."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the submission as JSON."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" not in arguments:
        args = parser.parse_args(arguments)
        parser.error("a command is required after HOST --")
    separator = arguments.index("--")
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1 :]
    if not command:
        parser.error("a command is required after HOST --")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if "\n" in args.cwd or "\n" in args.log:
        parser.error("--cwd and --log cannot contain newlines")

    try:
        hosts = read_nodelist(args.nodelist)
    except ValueError as exc:
        parser.error(str(exc))
    if args.host not in hosts:
        parser.error(
            f"host {args.host!r} is not present in nodelist {args.nodelist.expanduser()}"
        )

    try:
        submission = submit_job(
            args.host,
            command,
            cwd=args.cwd,
            log=args.log,
            timeout=args.timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(submission_json(submission))
    else:
        print(f"Submitted {submission.job_id} to {submission.host}")
        print(f"Local SSH PID: {submission.pid}")
        print(f"Remote working directory: {submission.cwd}")
        print(f"Local log: {submission.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
