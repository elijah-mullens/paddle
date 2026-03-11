from __future__ import annotations

import argparse
from typing import Sequence

from .cubed_sphere_remap import main as cubed_sphere_remap_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paddle")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "cs-remap",
        help="Remap stitched cubed-sphere NetCDF outputs to a lat-lon grid.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command == "cs-remap":
        return cubed_sphere_remap_main(remaining)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
