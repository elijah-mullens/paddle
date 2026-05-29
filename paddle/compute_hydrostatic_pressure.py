#!/usr/bin/env python3
"""
ECMWF data fetching and curation pipeline - Step 4.

This script computes a hydrostatically balanced pressure field at cell centers
from the regridded data produced in Step 3.

The script performs the following steps:
1. Parse YAML configuration to extract gravity from forcing.const-gravity.grav1
2. Load the regridded NetCDF file from Step 3
3. Integrate hydrostatic balance equation from top to bottom: pf[i-1] = pf[i] + rho[i] * grav
4. Compute cell center pressure using geometric mean: p[i] = sqrt(pf[i] * pf[i+1])
5. Augment the NetCDF file with the new variable 'p' at cell centers

Usage:
    python compute_hydrostatic_pressure.py <config.yaml> <regridded.nc>

    python compute_hydrostatic_pressure.py earth.yaml regridded_era5.nc

Requirements:
    - PyYAML for YAML parsing
    - xarray, netCDF4 for NetCDF I/O
    - numpy for numerical operations
"""

import argparse
import sys
import os
import yaml
from datetime import datetime, timezone
from typing import Dict

# Add current directory to path for importing local modules
sys.path.insert(0, os.path.dirname(__file__))


def parse_yaml_config(yaml_file: str) -> Dict:
    """
    Parse YAML configuration file.

    Args:
        yaml_file: Path to YAML configuration file

    Returns:
        Dictionary containing parsed YAML content

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Configuration file not found: {yaml_file}")

    with open(yaml_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file: {e}")

    return config


def extract_gravity(config: Dict, default: float = 9.80665) -> float:
    """
    Extract gravity value from configuration.

    Args:
        config: Parsed YAML configuration dictionary
        default: Default gravity value to use if not found in config (m/s^2)

    Returns:
        Gravity value (positive, in m/s^2). Returns default if not found.
    """
    try:
        if "forcing" in config:
            forcing = config["forcing"]
            if "const-gravity" in forcing:
                const_gravity = forcing["const-gravity"]
                if "grav1" in const_gravity:
                    grav1 = float(const_gravity["grav1"])
                    # grav1 is typically negative (pointing downward), convert to positive
                    gravity = abs(grav1)
                    if gravity > 0:
                        return gravity
    except (KeyError, TypeError, ValueError):
        pass

    # Return default if not found or parsing fails
    print(
        f"   Warning: Could not read gravity from config, using default: {default} m/s²"
    )
    return default


def load_regridded_data(nc_file: str):
    """
    Load regridded NetCDF data from Step 3.

    Args:
        nc_file: Path to regridded NetCDF file

    Returns:
        xarray Dataset containing the regridded data

    Raises:
        FileNotFoundError: If NetCDF file doesn't exist
        ValueError: If required variables are missing
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for loading NetCDF files. "
            "Install with: pip install xarray netCDF4"
        )

    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

    print(f"\nLoading regridded data from: {nc_file}")
    ds = xr.open_dataset(nc_file)

    # Check for required variables
    if "pressure_level" not in ds.data_vars:
        raise ValueError("NetCDF file must contain 'pressure_level' variable")

    if "rho" not in ds.data_vars:
        raise ValueError("NetCDF file must contain 'rho' variable")

    # Check for required coordinates
    if "x1f" not in ds.coords:
        raise ValueError(
            "NetCDF file must contain 'x1f' coordinate (vertical interfaces)"
        )

    if "x1" not in ds.coords:
        raise ValueError("NetCDF file must contain 'x1' coordinate (vertical centers)")

    print(f"  Found pressure_level: shape {ds['pressure_level'].shape}")
    print(f"  Found rho: shape {ds['rho'].shape}")
    print(f"  Vertical interfaces (x1f): {len(ds['x1f'])} levels")
    print(f"  Vertical centers (x1): {len(ds['x1'])} levels")

    return ds


def compute_hydrostatic_pressure(ds, gravity: float):
    """
    Compute hydrostatically balanced pressure at cell centers.

    Starting from the top level (highest) and integrating downward:
        pf[i-1] = pf[i] + rho[i] * grav * dz

    where:
        pf[i] is pressure at i-th interface (from pressure_level variable)
        rho[i] is density at i-th cell center
        dz is the vertical spacing between interfaces

    Then compute cell center pressure as geometric mean:
        p[i] = sqrt(pf[i] * pf[i+1])

    Args:
        ds: xarray Dataset with pressure_level and rho variables
        gravity: Gravity value (positive, m/s^2)

    Returns:
        numpy array of pressure at cell centers with shape (T, Z, Y, X)
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("NumPy is required. Install with: pip install numpy")

    print("\nComputing hydrostatically balanced pressure field...")

    # Get data arrays
    pressure_level = ds["pressure_level"].values  # Shape: (T, Zf, Y, X) at interfaces
    rho = ds["rho"].values  # Shape: (T, Z, Y, X) at cell centers
    x1f = ds["x1f"].values  # Vertical coordinates at interfaces

    T, Zf, Y, X = pressure_level.shape
    Z = Zf - 1  # Number of cell centers

    # Verify rho shape
    if rho.shape != (T, Z, Y, X):
        raise ValueError(
            f"Density shape {rho.shape} doesn't match expected ({T}, {Z}, {Y}, {X})"
        )

    print(f"  Processing {T} time steps, {Z} vertical cells, {Y}x{X} horizontal grid")

    # Compute dz for each cell (distance between interfaces)
    dz = np.diff(x1f)  # Shape: (Z,)

    # Initialize pressure at interfaces (will be recomputed from hydrostatic balance)
    pf_new = np.zeros((T, Zf, Y, X), dtype=np.float64)

    # Start from the top interface (highest altitude, index Zf-1)
    # Use the original pressure_level at the top
    pf_new[:, -1, :, :] = pressure_level[:, -1, :, :]

    print(f"  Starting from top interface (z = {x1f[-1]:.1f} m)")
    print(
        f"  Top pressure range: [{np.nanmin(pf_new[:, -1, :, :]):.1f}, "
        f"{np.nanmax(pf_new[:, -1, :, :]):.1f}] Pa"
    )

    # Integrate downward: pf[i] = pf[i+1] + rho[i] * g * dz[i]
    # Note: In the array, lower altitude has lower index
    # So we integrate from high index (top) to low index (bottom)
    for i in range(Z - 1, -1, -1):
        # pf[i] = pf[i+1] + rho[i] * gravity * dz[i]
        # where i is the interface below cell i, and i+1 is the interface above cell i
        pf_new[:, i, :, :] = pf_new[:, i + 1, :, :] + rho[:, i, :, :] * gravity * dz[i]

    print(
        f"  Bottom pressure range: [{np.nanmin(pf_new[:, 0, :, :]):.1f}, "
        f"{np.nanmax(pf_new[:, 0, :, :]):.1f}] Pa"
    )

    # Compute cell center pressure as geometric mean
    # p[i] = sqrt(pf[i] * pf[i+1])
    # Use float32 to match NetCDF storage format and maintain consistency with other variables
    # Precision loss is negligible (< 1e-6 relative error) for typical pressure values
    p = np.zeros((T, Z, Y, X), dtype=np.float32)

    for i in range(Z):
        # Cell i is bounded by interfaces i (below) and i+1 (above)
        p[:, i, :, :] = np.sqrt(pf_new[:, i, :, :] * pf_new[:, i + 1, :, :])

    print(f"  Cell center pressure computed")
    print(f"  Pressure range: [{np.nanmin(p):.1f}, {np.nanmax(p):.1f}] Pa")

    # Check for NaN or invalid values
    n_nan = np.sum(np.isnan(p))
    n_neg = np.sum(p < 0)

    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN values in pressure field")
    if n_neg > 0:
        print(f"  WARNING: {n_neg} negative values in pressure field")

    return p


def augment_netcdf_with_pressure(nc_file: str, pressure: "np.ndarray") -> None:
    """
    Augment the NetCDF file with the new pressure variable at cell centers.

    Args:
        nc_file: Path to NetCDF file to augment
        pressure: Pressure array with shape (T, Z, Y, X)
    """
    try:
        from netCDF4 import Dataset
        import numpy as np
    except ImportError:
        raise ImportError("netCDF4 is required. Install with: pip install netCDF4")

    print(f"\nAugmenting NetCDF file with pressure variable...")
    print(f"  File: {nc_file}")

    # Open NetCDF file in append mode
    with Dataset(nc_file, "a") as ncfile:
        # Check if 'p' already exists, remove it if so
        if "p" in ncfile.variables:
            print("  Removing existing 'p' variable...")
            # Cannot delete variables in NetCDF4, need to warn user
            print(
                "  WARNING: 'p' variable already exists. NetCDF format doesn't support deletion."
            )
            print("  The variable will be overwritten.")
            # Actually we can overwrite by just assigning new values
        else:
            # Create new variable 'p' at cell centers
            print("  Creating new variable 'p'...")
            p_var = ncfile.createVariable(
                "p", "f4", ("time", "x1", "x2", "x3"), zlib=True, complevel=4
            )

            # Set attributes
            p_var.units = "Pa"
            p_var.long_name = "Hydrostatically balanced pressure at cell centers"
            p_var.standard_name = "air_pressure"
            p_var.description = (
                "Pressure computed from hydrostatic balance integration using "
                "density and pressure_level, then calculated at cell centers "
                "using geometric mean of interface pressures"
            )

        # Write data
        ncfile.variables["p"][:] = pressure.astype("f4")

        # Update global attributes
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Append to history
        if "history" in ncfile.ncattrs():
            existing_history = ncfile.history
            ncfile.history = (
                f"{existing_history}\n"
                f"{current_time}: Added hydrostatically balanced pressure 'p' at cell centers "
                "computed from hydrostatic integration and geometric mean."
            )
        else:
            ncfile.history = f"{current_time}: Added hydrostatically balanced pressure 'p' at cell centers."

        print("  Variable 'p' successfully added to NetCDF file")
        print(f"  Shape: {pressure.shape}")
        print(f"  Data type: float32")


def compute_hydrostatic_pressure_pipeline(config_file: str, nc_file: str) -> None:
    """
    Main pipeline function for Step 4.

    Args:
        config_file: Path to YAML configuration file
        nc_file: Path to regridded NetCDF file from Step 3
    """
    print("=" * 70)
    print("ECMWF Data Curation - Step 4: Compute Hydrostatic Pressure")
    print("=" * 70)

    # Step 1: Parse configuration and extract gravity
    print(f"\n1. Reading configuration from: {config_file}")
    config = parse_yaml_config(config_file)
    gravity = extract_gravity(config)
    print(f"   Gravity: {gravity:.6f} m/s²")

    # Step 2: Load regridded data
    print(f"\n2. Loading regridded NetCDF file: {nc_file}")
    ds = load_regridded_data(nc_file)

    # Step 3: Compute hydrostatically balanced pressure
    print(f"\n3. Computing hydrostatic pressure field...")
    pressure = compute_hydrostatic_pressure(ds, gravity)

    # Close the dataset before modifying the file
    ds.close()

    # Step 4: Augment NetCDF file
    print(f"\n4. Augmenting NetCDF file with pressure variable...")
    augment_netcdf_with_pressure(nc_file, pressure)

    print("\n" + "=" * 70)
    print("Step 4 completed successfully!")
    print("=" * 70)
    print(f"\nOutput file: {nc_file}")
    print(f"New variable: 'p' (pressure at cell centers)")
    print(f"Shape: {pressure.shape}")


def main():
    """Main function to execute hydrostatic pressure computation."""
    parser = argparse.ArgumentParser(
        description="ECMWF data curation pipeline - Step 4: Compute hydrostatic pressure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script computes hydrostatically balanced pressure at cell centers from
the regridded data produced in Step 3.

Example:
    python compute_hydrostatic_pressure.py earth.yaml regridded_era5.nc

The script will:
  1. Read gravity from forcing.const-gravity.grav1 in YAML config
  2. Load pressure_level (at interfaces) and rho (at centers) from NetCDF
  3. Integrate hydrostatic balance from top to bottom
  4. Compute cell center pressure using geometric mean
  5. Augment NetCDF file with new variable 'p'
        """,
    )

    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "nc_file", type=str, help="Path to regridded NetCDF file from Step 3"
    )

    args = parser.parse_args()

    try:
        compute_hydrostatic_pressure_pipeline(args.config_file, args.nc_file)

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"\n✗ Configuration/Data error: {e}")
        sys.exit(1)

    except yaml.YAMLError as e:
        print(f"\n✗ YAML parsing error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
