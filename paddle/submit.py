from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

from .nodes import DEFAULT_NODELIST, read_nodelist


REMOTE_SUBMIT = r"""
set -eu

cwd=$1
log=$2
shift 2

case "$cwd" in
  "~") cwd=$HOME ;;
  "~/"*) cwd=$HOME/${cwd#~/} ;;
esac
case "$log" in
  "~") log=$HOME ;;
  "~/"*) log=$HOME/${log#~/} ;;
esac

if ! cd -- "$cwd"; then
  echo "cannot use remote working directory: $cwd" >&2
  exit 3
fi
cwd=$(pwd -P)
job_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
if [ -z "$log" ]; then
  log=$HOME/.local/state/paddle/jobs/$job_id.log
fi
case "$log" in
  /*) ;;
  *) log=$cwd/$log ;;
esac
log_dir=$(dirname -- "$log")
if ! mkdir -p -- "$log_dir"; then
  echo "cannot create remote log directory: $log_dir" >&2
  exit 4
fi

nohup "$@" >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "__PADDLE_SUBMIT__" "$job_id" "$pid" "$cwd" "$log"
"""


@dataclass
class Submission:
    job_id: str
    host: str
    pid: int
    cwd: str
    log: str
    command: list[str]


def _remote_invocation(cwd: str, log: str, command: Sequence[str]) -> str:
    arguments = [cwd, log, *command]
    return "sh -s -- " + " ".join(shlex.quote(argument) for argument in arguments)


def _last_error(stderr: str, fallback: str) -> str:
    lines = stderr.strip().splitlines()
    return lines[-1] if lines else fallback


def submit_job(
    host: str,
    command: Sequence[str],
    *,
    cwd: str = "~",
    log: str = "",
    timeout: float = 10.0,
) -> Submission:
    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        "--",
        host,
        _remote_invocation(cwd, log, command),
    ]
    try:
        result = subprocess.run(
            ssh_command,
            input=REMOTE_SUBMIT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"SSH submission timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise ValueError(str(exc)) from exc

    if result.returncode != 0:
        raise ValueError(_last_error(result.stderr, f"SSH exited {result.returncode}"))

    lines = result.stdout.splitlines()
    try:
        marker = len(lines) - 1 - lines[::-1].index("__PADDLE_SUBMIT__")
        job_id, pid, remote_cwd, remote_log = lines[marker + 1 : marker + 5]
        if not job_id or not remote_cwd or not remote_log:
            raise ValueError
        remote_pid = int(pid)
    except (ValueError, IndexError):
        raise ValueError(
            "remote host returned an invalid submission response"
        ) from None

    return Submission(job_id, host, remote_pid, remote_cwd, remote_log, list(command))


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
        help="Remote log path (default: generated under ~/.local/state/paddle/jobs).",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="SSH startup timeout in seconds."
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
        print(
            f"Submitted {submission.job_id} to {submission.host} (PID {submission.pid})"
        )
        print(f"Working directory: {submission.cwd}")
        print(f"Log: {submission.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
