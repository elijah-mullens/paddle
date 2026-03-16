import argparse

from paddle import find_init_params, setup_profile, write_profile
from snapy import MeshBlock, MeshBlockOptions


def setup_saturn_profile(infile: str, output: str) -> None:
    options = MeshBlockOptions.from_yaml(infile, verbose=False)
    block = MeshBlock(options)

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
    hydro_w = setup_profile(block, param, method=method, verbose=False)
    write_profile(output, block, hydro_w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="saturn1d.yaml")
    parser.add_argument("--output", default="saturn_profile.txt")
    args = parser.parse_args()
    setup_saturn_profile(args.input, args.output)


if __name__ == "__main__":
    main()
