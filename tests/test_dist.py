from __future__ import annotations

from types import SimpleNamespace

import torch

from paddle import dist as paddle_dist


def test_start_dist_registers_commux_for_ucx_cpu(monkeypatch) -> None:
    calls: list[object] = []
    fake_commux = SimpleNamespace(register=lambda: calls.append("register"))
    monkeypatch.setitem(__import__("sys").modules, "commux", fake_commux)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        paddle_dist.dist,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        paddle_dist.dist_c10d,
        "_get_default_group",
        lambda: "group",
    )
    monkeypatch.setattr(
        paddle_dist.snapy.distributed,
        "set_process_group",
        lambda group: calls.append(("snapy", group)),
    )

    device = paddle_dist.start_dist("ucx")

    assert device == torch.device("cpu")
    assert calls == [
        "register",
        ("init", {"backend": "ucx", "init_method": "env://"}),
        ("snapy", "group"),
    ]


def test_start_dist_registers_commux_for_ucx_cuda(monkeypatch) -> None:
    calls: list[object] = []
    fake_commux = SimpleNamespace(register=lambda: calls.append("register"))
    monkeypatch.setitem(__import__("sys").modules, "commux", fake_commux)
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda device: calls.append(("cuda", device))
    )
    monkeypatch.setattr(
        paddle_dist.dist,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        paddle_dist.dist_c10d,
        "_get_default_group",
        lambda: "group",
    )
    monkeypatch.setattr(
        paddle_dist.snapy.distributed,
        "set_process_group",
        lambda group: calls.append(("snapy", group)),
    )

    device = paddle_dist.start_dist("ucx")

    assert device == torch.device("cuda:3")
    assert calls == [
        "register",
        ("cuda", 3),
        ("init", {"backend": "ucx", "init_method": "env://"}),
        ("snapy", "group"),
    ]
