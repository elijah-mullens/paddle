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
    submission_json,
    submit_job,
    transfer_files,
)


def test_remote_invocation_preserves_literal_arguments() -> None:
    invocation = _remote_invocation("~/work dir", ["printf", "%s", "hello; $(false)"])
    arguments = shlex.split(invocation)
    assert arguments[:2] == ["bash", "-c"]
    assert arguments[3:] == [
        "bash",
        "~/work dir",
        "printf",
        "%s",
        "hello; $(false)",
    ]
    assert 'source "${HOME}/.bash_profile"' in arguments[2]
    assert "shopt -s expand_aliases" in arguments[2]
    assert "printf -v command_line '%q ' \"$@\"" in arguments[2]
    assert 'eval "$command_line"' in arguments[2]


def test_remote_invocation_expands_alias_from_bash_profile(tmp_path: Path) -> None:
    (tmp_path / ".bash_profile").write_text(
        "alias paddle_test_alias='printf alias-expanded'\n", encoding="utf-8"
    )
    invocation = _remote_invocation("~", ["paddle_test_alias"])

    result = subprocess.run(
        ["bash", "-c", invocation],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "alias-expanded"


def test_submit_job_streams_to_local_log(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []
    transfers: list[tuple[str, list[Path], str, float]] = []

    class Process:
        pid = 123

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("ssh", timeout)

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
        log=str(tmp_path / "job.log"),
        timeout=7,
        files=[source],
    )

    assert transfers == [("dart1", [source], "~/work", 7)]
    assert submission == Submission(
        "job-1",
        "dart1",
        123,
        "~/work",
        str(tmp_path / "job.log"),
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
    assert shlex.split(ssh_command[7])[-4:] == [
        "python",
        "train.py",
        "--epochs",
        "2",
    ]
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    assert options["start_new_session"] is True
    assert options["stdout"].name == str(tmp_path / "job.log")


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


def test_submit_job_defaults_log_to_current_local_directory(
    tmp_path: Path, monkeypatch
) -> None:
    class Process:
        pid = 123

        def wait(self, timeout):
            return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("paddle.submit._job_id", lambda: "job-1")
    monkeypatch.setattr(
        "paddle.submit.subprocess.Popen", lambda *args, **kwargs: Process()
    )

    submission = submit_job("dart1", ["true"])
    assert submission.log == str(tmp_path / "job-1.log")
    assert (tmp_path / "job-1.log").exists()


def test_submit_job_reports_start_failure(tmp_path: Path, monkeypatch) -> None:
    def missing_ssh(*args, **kwargs):
        raise OSError("ssh missing")

    monkeypatch.setattr("paddle.submit.subprocess.Popen", missing_ssh)
    with pytest.raises(ValueError, match="ssh missing"):
        submit_job("dart1", ["true"], log=str(tmp_path / "missing.log"))

    class FailedProcess:
        pid = 123

        def wait(self, timeout):
            return 255

    def failed_ssh(*args, **kwargs):
        kwargs["stdout"].write("connection refused\n")
        kwargs["stdout"].flush()
        return FailedProcess()

    monkeypatch.setattr("paddle.submit.subprocess.Popen", failed_ssh)
    with pytest.raises(ValueError, match="connection refused"):
        submit_job("dart1", ["true"], log=str(tmp_path / "failed.log"))


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
