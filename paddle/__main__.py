from __future__ import annotations

import argparse
from typing import Sequence


def cubed_sphere_shrink_main(argv: Sequence[str] | None = None) -> int:
    from .cubed_sphere_shrink import main

    return main(argv)


def cubed_sphere_expand_main(argv: Sequence[str] | None = None) -> int:
    from .cubed_sphere_expand import main

    return main(argv)


def cubed_sphere_remap_main(argv: Sequence[str] | None = None) -> int:
    from .cubed_sphere_remap import main

    return main(argv)


def monitor_main(argv: Sequence[str] | None = None) -> int:
    from .monitor import main

    return main(argv)


def submit_main(argv: Sequence[str] | None = None) -> int:
    from .submit import main

    return main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paddle")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "cs-shrink",
        add_help=False,
        help="Shrink a 6*N^2-block cubed-sphere restart into six face blocks.",
    )
    subparsers.add_parser(
        "cs-expand",
        add_help=False,
        help="Expand six cubed-sphere face blocks into 6*N^2 blocks.",
    )
    subparsers.add_parser(
        "cs-remap",
        help="Remap stitched cubed-sphere NetCDF outputs to a lat-lon grid.",
    )
    subparsers.add_parser(
        "monitor",
        add_help=False,
        help="Monitor CPU, GPU, and disk usage on machines over SSH.",
    )
    subparsers.add_parser(
        "submit",
        add_help=False,
        help="Run a command on a machine in the nodelist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command == "cs-shrink":
        return cubed_sphere_shrink_main(remaining)
    if args.command == "cs-expand":
        return cubed_sphere_expand_main(remaining)
    if args.command == "cs-remap":
        return cubed_sphere_remap_main(remaining)
    if args.command == "monitor":
        return monitor_main(remaining)
    if args.command == "submit":
        return submit_main(remaining)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
