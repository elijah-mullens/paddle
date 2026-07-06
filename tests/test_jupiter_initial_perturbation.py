from __future__ import annotations

import math

import pytest
import torch

import spherical_random
from spherical_random import randomize_initial_velocity


class FakeCoord:
    def __init__(self, x1v: torch.Tensor, x2v: torch.Tensor, x3v: torch.Tensor):
        self._buffers = {"x1v": x1v, "x2v": x2v, "x3v": x3v}

    def buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]


class FakeLayoutOptions:
    def __init__(self, rank: int = 0):
        self._rank = rank

    def rank(self) -> int:
        return self._rank


class FakeLayout:
    def __init__(self, loc: tuple[int, ...]):
        self.options = FakeLayoutOptions()
        self._loc = loc

    def loc_of(self, rank: int) -> tuple[int, ...]:
        del rank
        return self._loc


class FakeBlock:
    def __init__(
        self,
        loc: tuple[int, ...],
        x1v: torch.Tensor,
        x2v: torch.Tensor,
        x3v: torch.Tensor,
    ):
        self._coord = FakeCoord(x1v, x2v, x3v)
        self._layout = FakeLayout(loc)

    def module(self, name: str):
        if name != "coord":
            raise KeyError(name)
        return self._coord

    def get_layout(self) -> FakeLayout:
        return self._layout


def make_block(face_id: int = 0) -> FakeBlock:
    x1v = torch.linspace(10.0, 20.0, 4, dtype=torch.float64)
    x2v = torch.linspace(-0.25 * math.pi, 0.25 * math.pi, 5, dtype=torch.float64)
    x3v = torch.linspace(-0.25 * math.pi, 0.25 * math.pi, 6, dtype=torch.float64)
    return FakeBlock((0, 0, face_id), x1v, x2v, x3v)


def test_cubed_sphere_perturbation_matches_target_metadata() -> None:
    target = torch.zeros((6, 5, 4), dtype=torch.float64)
    perturbation = randomize_initial_velocity(make_block(), target)

    assert perturbation.shape == target.shape
    assert perturbation.dtype == target.dtype
    assert perturbation.device == target.device
    assert torch.all(perturbation >= 0.0)
    assert torch.all(perturbation <= 0.1)


def test_cubed_sphere_perturbation_is_reproducible() -> None:
    target = torch.zeros((6, 5, 4), dtype=torch.float64)
    block = make_block()

    first = randomize_initial_velocity(block, target)
    second = randomize_initial_velocity(block, target)

    torch.testing.assert_close(first, second)


def test_cubed_sphere_perturbation_samples_one_global_field_across_faces() -> None:
    target = torch.zeros((6, 5, 4), dtype=torch.float64)

    face0 = randomize_initial_velocity(make_block(face_id=0), target)
    face1 = randomize_initial_velocity(make_block(face_id=1), target)

    assert not torch.equal(face0, face1)


def test_cubed_sphere_perturbation_wraps_longitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_lonlat(face: str, alpha: torch.Tensor, beta: torch.Tensor):
        del face, beta
        lon = torch.where(
            alpha < 0.0,
            torch.zeros_like(alpha),
            torch.full_like(alpha, 2.0 * math.pi),
        )
        lat = torch.zeros_like(alpha)
        return lon, lat

    monkeypatch.setattr(spherical_random, "cs_ab_to_lonlat", fake_lonlat)
    target = torch.zeros((2, 2, 3), dtype=torch.float64)
    block = FakeBlock(
        (0, 0, 0),
        torch.linspace(1.0, 3.0, 3, dtype=torch.float64),
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
    )

    perturbation = randomize_initial_velocity(block, target)

    torch.testing.assert_close(perturbation[:, 0, :], perturbation[:, 1, :])


def test_non_cubed_sphere_uses_rand_like(monkeypatch: pytest.MonkeyPatch) -> None:
    target = torch.zeros((2, 3, 4), dtype=torch.float64)
    calls = []

    def fake_rand_like(tensor: torch.Tensor) -> torch.Tensor:
        calls.append(tensor)
        return torch.full_like(tensor, 0.25)

    monkeypatch.setattr(spherical_random.torch, "rand_like", fake_rand_like)

    perturbation = randomize_initial_velocity(
        FakeBlock((0, 0), torch.arange(4.0), torch.arange(3.0), torch.arange(2.0)),
        target,
        amplitude=0.2,
    )

    assert calls == [target]
    torch.testing.assert_close(perturbation, torch.full_like(target, 0.05))
