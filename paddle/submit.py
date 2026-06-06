from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Optional, Sequence, Union
from uuid import uuid4

from .nodes import DEFAULT_NODELIST, read_nodelist


REMOTE_RUN = r"""
cwd=$1
job_id=$2
shift 2

shopt -s expand_aliases
if ! source "${HOME}/.bash_profile"; then
  echo "cannot source remote bash profile: ${HOME}/.bash_profile" >&2
  exit 2
fi
set -eu
export PADDLE_JOB_ID=$job_id

case "$cwd" in
  "~") cwd=$HOME ;;
  "~/"*) cwd=$HOME/${cwd#~/} ;;
esac

if ! cd -- "$cwd"; then
  echo "cannot use remote working directory: $cwd" >&2
  exit 3
fi

printf -v command_line '%q ' "$@"
set -m
eval "$command_line" &
pid=$!

collect_descendants() {
  local parent=$1
  local child
  for child in $(ps -o pid= --ppid "$parent" 2>/dev/null); do
    collect_descendants "$child"
    printf '%s\n' "$child"
  done
}

collect_tagged_processes() {
  local environment
  local process
  for environment in /proc/[0-9]*/environ; do
    if grep -Fzq "PADDLE_JOB_ID=$job_id" "$environment" 2>/dev/null; then
      process=${environment#/proc/}
      process=${process%/environ}
      if [ "$process" != "$$" ]; then
        printf '%s\n' "$process"
      fi
    fi
  done
}

terminate_job() {
  trap - HUP INT TERM
  targets=$(printf '%s\n%s\n' "$(collect_descendants "$pid")" "$(collect_tagged_processes)" | sort -un)
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  if [ -n "$targets" ]; then
    kill -TERM $targets 2>/dev/null || true
  fi
  sleep 1
  targets=$(printf '%s\n%s\n' "$targets" "$(collect_tagged_processes)" | sort -un)
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  if [ -n "$targets" ]; then
    kill -KILL $targets 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  exit 130
}
trap terminate_job HUP INT TERM

set +e
wait "$pid"
status=$?
trap - HUP INT TERM
exit "$status"
"""

REMOTE_SUBMIT = r"""
set -eu

cwd=$1
log=$2
job_id=$3
runner=$4
shift 4

case "$cwd" in
  "~") cwd=$HOME ;;
  "~/"*) cwd=$HOME/${cwd#~/} ;;
esac
case "$log" in
  "~") log=$HOME ;;
  "~/"*) log=$HOME/${log#~/} ;;
  /*) ;;
  *) log=$cwd/$log ;;
esac

if ! cd -- "$cwd"; then
  echo "cannot use remote working directory: $cwd" >&2
  exit 3
fi
cwd=$(pwd -P)
log_dir=$(dirname -- "$log")
if ! mkdir -p -- "$log_dir"; then
  echo "cannot create remote log directory: $log_dir" >&2
  exit 4
fi

nohup bash -c "$runner" bash "$cwd" "$job_id" "$@" >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "__PADDLE_SUBMIT__" "$job_id" "$pid" "$cwd" "$log"
"""

REMOTE_RECEIVE_FILE = r"""
set -eu

cwd=$1
name=$2
mode=$3

case "$cwd" in
  "~") cwd=$HOME ;;
  "~/"*) cwd=$HOME/${cwd#~/} ;;
esac

if ! cd -- "$cwd"; then
  echo "cannot use remote working directory: $cwd" >&2
  exit 3
fi

temp=".paddle-transfer.$$"
trap 'rm -f -- "$temp"' EXIT
cat > "$temp"
chmod "$mode" "$temp"
mv -f -- "$temp" "$name"
trap - EXIT
"""


@dataclass
class Submission:
    job_id: str
    host: str
    pid: Optional[int]
    cwd: str
    log: Optional[str]
    command: list[str]


def _bash_invocation(script: str, arguments: Sequence[str]) -> str:
    positional = ["bash", *arguments]
    return "bash -c " + " ".join(
        shlex.quote(argument) for argument in [script, *positional]
    )


def _remote_invocation(cwd: str, job_id: str, command: Sequence[str]) -> str:
    return _bash_invocation(REMOTE_RUN, [cwd, job_id, *command])


def _last_error(stderr: Union[str, bytes], fallback: str) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    lines = stderr.strip().splitlines()
    return lines[-1] if lines else fallback


def _job_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _ssh_command(host: str, timeout: float, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        "--",
        host,
        remote_command,
    ]


def transfer_files(
    host: str,
    files: Sequence[Path],
    *,
    cwd: str,
    timeout: float,
) -> None:
    destinations: set[str] = set()
    sources: list[tuple[Path, str]] = []
    for raw_path in files:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"transfer source is not a regular file: {raw_path}")
        if path.name in destinations:
            raise ValueError(f"multiple transfer files have basename {path.name!r}")
        destinations.add(path.name)
        sources.append((path, f"{path.stat().st_mode & 0o777:o}"))

    for path, mode in sources:
        remote_command = _bash_invocation(REMOTE_RECEIVE_FILE, [cwd, path.name, mode])
        try:
            with path.open("rb") as source:
                result = subprocess.run(
                    _ssh_command(host, timeout, remote_command),
                    stdin=source,
                    capture_output=True,
                    check=False,
                )
        except OSError as exc:
            raise ValueError(f"cannot transfer {path}: {exc}") from exc
        if result.returncode != 0:
            lines = result.stderr.decode(errors="replace").strip().splitlines()
            error = lines[-1] if lines else f"SSH exited {result.returncode}"
            raise ValueError(f"cannot transfer {path}: {error}")


def run_foreground(
    host: str, command: Sequence[str], *, cwd: str, job_id: str, timeout: float
) -> int:
    try:
        process = subprocess.Popen(
            _ssh_command(host, timeout, _remote_invocation(cwd, job_id, command))
        )
    except OSError as exc:
        raise ValueError(str(exc)) from exc

    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def submit_job(
    host: str,
    command: Sequence[str],
    *,
    cwd: str = "~",
    log: str = "",
    timeout: float = 10.0,
    files: Sequence[Path] = (),
) -> Submission:
    transfer_files(host, files, cwd=cwd, timeout=timeout)

    job_id = _job_id()
    if not log:
        returncode = run_foreground(
            host, command, cwd=cwd, job_id=job_id, timeout=timeout
        )
        if returncode != 0:
            raise ValueError(f"remote command exited {returncode}")
        return Submission(job_id, host, None, cwd, None, list(command))

    remote_command = _bash_invocation(
        REMOTE_SUBMIT, [cwd, log, job_id, REMOTE_RUN, *command]
    )
    try:
        result = subprocess.run(
            _ssh_command(host, timeout, remote_command),
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
        receipt_job_id, pid, remote_cwd, remote_log = lines[marker + 1 : marker + 5]
        if receipt_job_id != job_id or not remote_cwd or not remote_log:
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
        description="Run a command on a machine in the nodelist.",
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
        help="Remote log path; when omitted, stream output to the terminal.",
    )
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        default=[],
        help="Local file to copy into the remote working directory; repeatable.",
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
            files=args.file,
        )
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(submission_json(submission))
    elif submission.log is None:
        print(f"Completed {submission.job_id} on {submission.host}")
    else:
        print(f"Submitted {submission.job_id} to {submission.host}")
        print(f"Remote PID: {submission.pid}")
        print(f"Remote working directory: {submission.cwd}")
        print(f"Remote log: {submission.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
