"""Run the moist Earth cloud-resolving example with the bundled earth_moist.yaml deck."""

from paddle.crm import main, make_params, run_with

__all__ = ["main", "make_params", "run_with"]


if __name__ == "__main__":
    main()
