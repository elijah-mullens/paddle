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
)


def test_remote_invocation_preserves_literal_arguments() -> None:
    invocation = _remote_invocation(
        "~/work dir", "~/job log.txt", ["printf", "%s", "hello; $(false)"]
    )
    assert shlex.split(invocation) == [
        "sh",
        "-s",
        "--",
        "~/work dir",
        "~/job log.txt",
        "printf",
        "%s",
        "hello; $(false)",
    ]


def test_submit_job_returns_remote_receipt(monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            "login banner\n__PADDLE_SUBMIT__\njob-1\n123\n/home/alice/work\n"
            "/home/alice/.local/state/paddle/jobs/job-1.log\n",
            "",
        )

    monkeypatch.setattr("paddle.submit.subprocess.run", fake_run)
    submission = submit_job(
        "dart1", ["python", "train.py", "--epochs", "2"], cwd="~/work", timeout=7
    )

    assert submission == Submission(
        "job-1",
        "dart1",
        123,
        "/home/alice/work",
        "/home/alice/.local/state/paddle/jobs/job-1.log",
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
    assert options["timeout"] == 7
    assert "nohup" in options["input"]


def test_submit_job_reports_timeout_ssh_failure_and_invalid_response(
    monkeypatch,
) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 2)

    monkeypatch.setattr("paddle.submit.subprocess.run", timeout)
    with pytest.raises(ValueError, match="timed out after 2s"):
        submit_job("dart1", ["true"], timeout=2)

    monkeypatch.setattr(
        "paddle.submit.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["ssh"], 255, "", "banner\nconnection refused\n"
        ),
    )
    with pytest.raises(ValueError, match="connection refused"):
        submit_job("dart1", ["true"])

    monkeypatch.setattr(
        "paddle.submit.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(["ssh"], 0, "hello\n", ""),
    )
    with pytest.raises(ValueError, match="invalid submission response"):
        submit_job("dart1", ["true"])


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
            {"cwd": "~/work", "log": "", "timeout": 10.0},
        )
    ]
    assert json.loads(capsys.readouterr().out)["job_id"] == "job-1"


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
