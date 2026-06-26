from pathlib import Path

import kintera

from .setup_profile import setup_profile
from .write_profile import write_profile
from .find_init_params import find_init_params
from .evolve_kinetics import evolve_kinetics
from .cubed_sphere_remap import remap_cubed_sphere_files
from .dist import start_dist, close_dist


def _add_packaged_kintera_resources() -> None:
    data_dir = Path(kintera.__file__).parent / "data"
    if data_dir.is_dir():
        kintera.add_resource_directory(str(data_dir))


_add_packaged_kintera_resources()

__all__ = [
    "start_dist",
    "close_dist",
    "setup_profile",
    "write_profile",
    "find_init_params",
    "evolve_kinetics",
    "remap_cubed_sphere_files",
]
__version__ = "1.3.7"
