from paddle import (
    setup_profile,
    write_profile,
    find_init_params,
)
from snapy import (
    MeshBlockOptions,
    MeshBlock,
)
from kintera import ThermoX


def setup_saturn_profile():
    path = "saturn1d.yaml"
    print(f"Reading input file: {path}")

    op_block = MeshBlockOptions.from_yaml(str(path), verbose=False)
    block = MeshBlock(op_block)

    param = {
        "Ts": 800.0,
        "Ps": 100.0e5,
        "Tmin": 85.0,
        "xH2O": 4.91e-2,
        "xNH3": 3.52e-4,
        "xH2S": 8.08e-5,
        "grav": 10.44,
    }

    method = "pseudo-adiabat"
    # method = "moist-adiabat"
    # method = "dry-adiabat"

    param = find_init_params(
        block,
        param,
        target_T=134.0,
        target_P=1.0e5,
        method=method,
        max_iter=50,
        ftol=1.0e-2,
        verbose=False,
    )

    print("Found parameters:")
    print(param)

    w = setup_profile(block, param, method=method, verbose=False)
    write_profile("saturn_profile.txt", block, w)
    return w


if __name__ == "__main__":
    w = setup_saturn_profile()
