"""evolve_kinetics at zero condensate (kintera >= b6ad373, #145).

kintera's Kinetics.jacobian needs the rate constants from
forward_with_rate_constants() when an evaporating solid has exactly zero
abundance; without them it raises "Evaporation at zero solid abundance requires
explicit rate_constants". A cloud deck starts that way, so the first kinetics
step of every evaporation deck died there. evolve_kinetics now passes kf.
"""

import pytest
import torch

kt = pytest.importorskip("kintera")
snapy = pytest.importorskip("snapy")
from paddle import evolve_kinetics  # noqa: E402

torch.set_default_dtype(torch.float64)

CARD = """
reference-state: {Tref: 0., Pref: 1.e5}
species:
  - {name: dry, composition: {H: 1.696, He: 0.152}, cv_R: 1.82}
  - {name: CH4, composition: {C: 1, H: 4}, cv_R: 3.0}
  - {name: "CH4(s)", composition: {C: 1, H: 4}, cv_R: 4.0, u0_R: -1166.9229}
reactions:
  - equation: CH4 <=> CH4(s)
    type: nucleation
    rate-constant: {formula: ideal, T3: 90.67, P3: 11690., beta: 12.87, gamma: 0.}
  - equation: CH4(s) => CH4
    type: evaporation
    rate-constant: {formula: ideal, T3: 90.67, P3: 11690., beta: 12.87, gamma: 0.,
                    diff_c: 0.2e-4, diff_T: 1.75, diff_P: -1., vm: 18.e-6, diameter: 1.0e-2}
"""


class _EOS:
    """Stands in for snapy's EquationOfState: evolve_kinetics only asks W->T."""

    def __init__(self, temp):
        self.temp = temp

    def compute(self, key, args):
        assert key == "W->T"
        return self.temp


def test_first_step_at_zero_condensate(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(CARD)
    thermo_y = kt.ThermoY(kt.ThermoOptions.from_yaml(str(path)))
    thermo_x = kt.ThermoX(kt.ThermoOptions.from_yaml(str(path)))
    kinet = kt.Kinetics(kt.KineticsOptions.from_yaml(str(path)))
    if not hasattr(kinet, "forward_with_rate_constants"):
        pytest.skip("kintera predates forward_with_rate_constants (#145)")

    nc = (1, 1, 2)
    ny = 2  # CH4, CH4(s)
    w = torch.zeros((snapy.kICY + ny,) + nc)
    w[snapy.kIDN] = 0.1
    w[snapy.kIPR] = 1.0e4
    w[snapy.kICY] = 1.0e-3  # vapour, subsaturated at 120 K
    w[snapy.kICY + 1] = 0.0  # NO condensate: the case that raised
    eos = _EOS(torch.full(nc, 120.0))

    del_rho = evolve_kinetics(w, eos, thermo_x, thermo_y, kinet, 1.0)
    assert del_rho.shape == (ny,) + nc
    assert torch.isfinite(del_rho).all()
