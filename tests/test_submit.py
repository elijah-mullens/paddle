from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess

import pytest

from paddle import __main__ as paddle_main
from paddle.submit import (
    Submission,
    _remote_invocation,
    main,
    run_foreground,
    submission_json,
    submit_job,
    transfer_files,
)


def test_remote_invocation_preserves_literal_arguments() -> None:
    invocation = _remote_invocation(
        "~/work dir", "job-1", ["printf", "%s", "hello; $(false)"]
    )
    arguments = shlex.split(invocation)
    assert arguments[:2] == ["bash", "-c"]
    assert arguments[3:] == [
        "bash",
        "~/work dir",
        "job-1",
        "printf",
        "%s",
        "hello; $(false)",
    ]
    assert 'source "${HOME}/.bash_profile"' in arguments[2]
    assert "shopt -s expand_aliases" in arguments[2]
    assert "printf -v command_line '%q ' \"$@\"" in arguments[2]
    assert 'eval "$command_line" &' in arguments[2]
    assert "trap terminate_job HUP INT TERM" in arguments[2]
    assert "collect_descendants" in arguments[2]
    assert "collect_tagged_processes" in arguments[2]
    assert "PADDLE_JOB_ID=$job_id" in arguments[2]
    assert 'kill -TERM -- "-$pid"' in arguments[2]
    assert 'kill -KILL -- "-$pid"' in arguments[2]


def test_remote_invocation_expands_alias_from_bash_profile(tmp_path: Path) -> None:
    (tmp_path / ".bash_profile").write_text(
        "alias paddle_test_alias='printf alias-expanded'\n", encoding="utf-8"
    )
    invocation = _remote_invocation("~", "job-1", ["paddle_test_alias"])

    result = subprocess.run(
        ["bash", "-c", invocation],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "alias-expanded"


def test_submit_job_without_log_streams_to_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[list[str], dict]] = []
    transfers: list[tuple[str, list[Path], str, float]] = []

    class Process:
        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr("paddle.submit.subprocess.Popen", fake_popen)
    monkeypatch.setattr("paddle.submit._job_id", lambda: "job-1")
    monkeypatch.setattr(
        "paddle.submit.transfer_files",
        lambda host, files, *, cwd, timeout: transfers.append(
            (host, list(files), cwd, timeout)
        ),
    )
    source = tmp_path / "input.yaml"
    submission = submit_job(
        "dart1",
        ["python", "train.py", "--epochs", "2"],
        cwd="~/work",
        timeout=7,
        files=[source],
    )

    assert transfers == [("dart1", [source], "~/work", 7)]
    assert submission == Submission(
        "job-1",
        "dart1",
        None,
        "~/work",
        None,
        ["python", "train.py", "--epochs", "2"],
    )
    ssh_command, options = calls[0]
    assert ssh_command[:6] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=7",
        "--",
    ]
    assert ssh_command[6] == "dart1"
    assert shlex.split(ssh_command[7])[-5:] == [
        "job-1",
        "python",
        "train.py",
        "--epochs",
        "2",
    ]
    assert options == {}


def test_run_foreground_terminates_ssh_on_keyboard_interrupt(monkeypatch) -> None:
    events: list[object] = []

    class Process:
        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            return -15

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(
        "paddle.submit.subprocess.Popen", lambda *args, **kwargs: Process()
    )

    with pytest.raises(KeyboardInterrupt):
        run_foreground("dart1", ["sleep", "60"], cwd="~", job_id="job-1", timeout=10)
    assert events == [("wait", None), "terminate", ("wait", 5)]


def test_run_foreground_kills_unresponsive_ssh_on_keyboard_interrupt(
    monkeypatch,
) -> None:
    events: list[object] = []

    class Process:
        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            if timeout == 5:
                raise subprocess.TimeoutExpired("ssh", timeout)
            return -9

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(
        "paddle.submit.subprocess.Popen", lambda *args, **kwargs: Process()
    )

    with pytest.raises(KeyboardInterrupt):
        run_foreground("dart1", ["sleep", "60"], cwd="~", job_id="job-1", timeout=10)
    assert events == [
        ("wait", None),
        "terminate",
        ("wait", 5),
        "kill",
        ("wait", None),
    ]


@pytest.mark.skipif(
    not Path("/usr/bin/setsid").exists(),
    reason="requires setsid to verify escaped descendant cleanup",
)
def test_remote_invocation_terminates_descendant_in_separate_session(
    tmp_path: Path,
) -> None:
    (tmp_path / ".bash_profile").write_text("", encoding="utf-8")
    pid_file = tmp_path / "escaped.pid"
    command = [
        "bash",
        "-c",
        f"setsid bash -c 'echo $$ > {pid_file}; sleep 60' & wait",
    ]
    wrapper = subprocess.Popen(
        ["bash", "-c", _remote_invocation("~", "job-1", command)],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    for _ in range(100):
        if pid_file.exists():
            break
        subprocess.run(["sleep", "0.02"], check=True)
    escaped_pid = int(pid_file.read_text())

    wrapper.terminate()
    wrapper.wait(timeout=5)
    for _ in range(100):
        if subprocess.run(["kill", "-0", str(escaped_pid)], check=False).returncode:
            break
        subprocess.run(["sleep", "0.02"], check=True)
    else:
        subprocess.run(["kill", "-9", str(escaped_pid)], check=False)
        pytest.fail(f"descendant {escaped_pid} survived wrapper termination")


def test_submit_job_with_log_launches_remote_background(monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            "banner\n__PADDLE_SUBMIT__\njob-1\n123\n/home/alice/work\n"
            "/mnt/jobs/output.log\n",
            "",
        )

    monkeypatch.setattr("paddle.submit.subprocess.run", fake_run)
    monkeypatch.setattr("paddle.submit._job_id", lambda: "job-1")
    submission = submit_job(
        "dart1",
        ["python", "train.py"],
        cwd="~/work",
        log="/mnt/jobs/output.log",
        timeout=7,
    )

    assert submission == Submission(
        "job-1",
        "dart1",
        123,
        "/home/alice/work",
        "/mnt/jobs/output.log",
        ["python", "train.py"],
    )
    remote_arguments = shlex.split(calls[0][0][-1])
    assert remote_arguments[-2:] == ["python", "train.py"]
    assert "/mnt/jobs/output.log" in remote_arguments
    assert "nohup bash -c" in remote_arguments[2]
    assert calls[0][1] == {
        "text": True,
        "capture_output": True,
        "timeout": 7,
        "check": False,
    }


def test_transfer_files_copies_contents_and_modes_to_remote_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "input.yaml"
    first.write_bytes(b"input: value\n")
    first.chmod(0o640)
    second = tmp_path / "run.sh"
    second.write_bytes(b"#!/bin/bash\ntrue\n")
    second.chmod(0o755)
    calls: list[tuple[list[str], dict, bytes]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs, kwargs["stdin"].read()))
        return subprocess.CompletedProcess(command, 0, b"", b"banner\n")

    monkeypatch.setattr("paddle.submit.subprocess.run", fake_run)
    transfer_files("dart1", [first, second], cwd="~/work", timeout=7)

    assert [contents for _, _, contents in calls] == [
        b"input: value\n",
        b"#!/bin/bash\ntrue\n",
    ]
    first_remote = shlex.split(calls[0][0][-1])
    second_remote = shlex.split(calls[1][0][-1])
    assert first_remote[-3:] == ["~/work", "input.yaml", "640"]
    assert second_remote[-3:] == ["~/work", "run.sh", "755"]
    assert 'mv -f -- "$temp" "$name"' in first_remote[2]


def test_transfer_files_rejects_invalid_duplicate_and_failed_sources(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(ValueError, match="not a regular file"):
        transfer_files("dart1", [tmp_path / "missing"], cwd="~", timeout=10)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "input.yaml"
    second = second_dir / "input.yaml"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    calls: list[list[str]] = []

    def successful_transfer(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(
        "paddle.submit.subprocess.run",
        successful_transfer,
    )
    with pytest.raises(ValueError, match="basename 'input.yaml'"):
        transfer_files("dart1", [first, second], cwd="~", timeout=10)
    assert calls == []

    monkeypatch.setattr(
        "paddle.submit.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, b"", b"permission denied\n"
        ),
    )
    with pytest.raises(ValueError, match="permission denied"):
        transfer_files("dart1", [first], cwd="~", timeout=10)


def test_submit_job_reports_run_and_submission_failures(monkeypatch) -> None:
    def missing_ssh(*args, **kwargs):
        raise OSError("ssh missing")

    monkeypatch.setattr("paddle.submit.subprocess.Popen", missing_ssh)
    with pytest.raises(ValueError, match="ssh missing"):
        submit_job("dart1", ["true"])

    monkeypatch.setattr("paddle.submit.run_foreground", lambda *args, **kwargs: 7)
    with pytest.raises(ValueError, match="remote command exited 7"):
        submit_job("dart1", ["false"])

    monkeypatch.setattr(
        "paddle.submit.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 255, "", "connection refused\n"
        ),
    )
    with pytest.raises(ValueError, match="connection refused"):
        submit_job("dart1", ["true"], log="job.log")

    monkeypatch.setattr(
        "paddle.submit.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "hello\n", ""),
    )
    with pytest.raises(ValueError, match="invalid submission response"):
        submit_job("dart1", ["true"], log="job.log")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 2)

    monkeypatch.setattr("paddle.submit.subprocess.run", timeout)
    with pytest.raises(ValueError, match="submission timed out after 2s"):
        submit_job("dart1", ["true"], log="job.log", timeout=2)


def test_submission_json_contains_receipt() -> None:
    result = Submission("job-1", "dart1", 123, "/work", "/log", ["true"])
    assert json.loads(submission_json(result)) == {
        "job_id": "job-1",
        "host": "dart1",
        "pid": 123,
        "cwd": "/work",
        "log": "/log",
        "command": ["true"],
    }


def test_submit_main_validates_nodelist_and_emits_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    nodelist = tmp_path / "nodes"
    nodelist.write_text("dart1\n", encoding="utf-8")
    calls: list[tuple[str, list[str], dict]] = []

    def fake_submit(host, command, **kwargs):
        calls.append((host, list(command), kwargs))
        return Submission("job-1", host, 123, "/work", "/log", list(command))

    monkeypatch.setattr("paddle.submit.submit_job", fake_submit)
    assert (
        main(
            [
                "dart1",
                "--cwd",
                "~/work",
                "--json",
                "--nodelist",
                str(nodelist),
                "--",
                "python",
                "train.py",
                "--epochs",
                "2",
            ]
        )
        == 0
    )
    assert calls == [
        (
            "dart1",
            ["python", "train.py", "--epochs", "2"],
            {"cwd": "~/work", "log": "", "timeout": 10.0, "files": []},
        )
    ]
    assert json.loads(capsys.readouterr().out)["job_id"] == "job-1"


def test_submit_main_defaults_to_remote_home(tmp_path: Path, monkeypatch) -> None:
    nodelist = tmp_path / "nodes"
    nodelist.write_text("dart1\n", encoding="utf-8")
    calls: list[dict] = []

    monkeypatch.setattr(
        "paddle.submit.submit_job",
        lambda host, command, **kwargs: calls.append(kwargs)
        or Submission("job-1", host, 123, "~", "/log", list(command)),
    )

    assert main(["dart1", "--nodelist", str(nodelist), "--", "true"]) == 0
    assert calls == [{"cwd": "~", "log": "", "timeout": 10.0, "files": []}]


def test_submit_main_accepts_repeated_files(tmp_path: Path, monkeypatch) -> None:
    nodelist = tmp_path / "nodes"
    nodelist.write_text("dart1\n", encoding="utf-8")
    first = tmp_path / "input.yaml"
    second = tmp_path / "run.py"
    calls: list[dict] = []

    monkeypatch.setattr(
        "paddle.submit.submit_job",
        lambda host, command, **kwargs: calls.append(kwargs)
        or Submission("job-1", host, 123, "~", "/log", list(command)),
    )

    assert (
        main(
            [
                "dart1",
                "--nodelist",
                str(nodelist),
                "--file",
                str(first),
                "--file",
                str(second),
                "--",
                "python",
                "run.py",
            ]
        )
        == 0
    )
    assert calls[0]["files"] == [first, second]


def test_submit_main_without_log_reports_completion(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    nodelist = tmp_path / "nodes"
    nodelist.write_text("dart1\n", encoding="utf-8")
    monkeypatch.setattr(
        "paddle.submit.submit_job",
        lambda host, command, **kwargs: Submission(
            "job-1", host, None, "~", None, list(command)
        ),
    )

    assert main(["dart1", "--nodelist", str(nodelist), "--", "true"]) == 0
    assert "Completed job-1 on dart1" in capsys.readouterr().out


def test_submit_main_returns_130_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch
) -> None:
    nodelist = tmp_path / "nodes"
    nodelist.write_text("dart1\n", encoding="utf-8")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("paddle.submit.submit_job", interrupt)
    assert main(["dart1", "--nodelist", str(nodelist), "--", "sleep", "60"]) == 130


def test_submit_main_rejects_unknown_host_missing_command_and_bad_timeout(
    tmp_path: Path,
) -> None:
    nodelist = tmp_path / "nodes"
    nodelist.write_text("dart1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["dart2", "--nodelist", str(nodelist), "--", "true"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["dart1", "--nodelist", str(nodelist)])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["dart1", "--timeout", "0", "--nodelist", str(nodelist), "--", "true"])
    assert exc.value.code == 2


def test_paddle_main_dispatches_submit(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        paddle_main, "submit_main", lambda args: calls.append(list(args)) or 7
    )
    assert paddle_main.main(["submit", "dart1", "--", "true"]) == 7
    assert calls == [["dart1", "--", "true"]]


def test_paddle_submit_help_is_forwarded(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        paddle_main.main(["submit", "--help"])
    assert exc.value.code == 0
    assert "--nodelist" in capsys.readouterr().out
