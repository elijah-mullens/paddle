# Cubed-Sphere Remapping

This guide shows how to remap Snapy stitched cubed-sphere NetCDF output files
to a regular latitude-longitude grid with `paddle` and TempestRemap.

The remapper is designed for the 2D stitched 3-by-2 mosaic that Snapy writes
for cubed-sphere runs. It handles:

- scalar fields such as `rho`, `press`, and `temp`
- vector triplets such as `vel1, vel2, vel3`

Vector fields are rotated to geographic components before remapping, so the
output file contains `vel_east`, `vel_north`, and `vel_up`.

## Requirements

Install the Python package in editable mode:

```bash
cd /home/chengcli/scix/repos/paddle
python -m pip install -e .
```

Install TempestRemap. The simplest option is conda:

```bash
conda install -c conda-forge tempest-remap
```

Or use the helper script in this repo:

```bash
cd /home/chengcli/scix/repos/paddle
./scripts/install_tempestremap.sh conda
```

Verify the required TempestRemap executables are available:

```bash
GenerateCSMesh --help
GenerateRLLMesh --help
GenerateOverlapMesh --help
GenerateOfflineMap --help
```

## Input Files

The remapper accepts one or more Snapy NetCDF files. For the current
sub-Neptune example, the inputs are:

```bash
/home/chengcli/scix/workspace/data/sub_neptune_tidallock.out1.00290.nc
/home/chengcli/scix/workspace/data/sub_neptune_tidallock.out2.00290.nc
```

`paddle` merges variables across those files into one remapped output. Duplicate
non-coordinate variable names across input files are treated as an error.

## Basic Usage

Use the dedicated script:

```bash
paddle-cs-remap \
  /home/chengcli/scix/workspace/data/sub_neptune_tidallock.out1.00290.nc \
  /home/chengcli/scix/workspace/data/sub_neptune_tidallock.out2.00290.nc \
  /home/chengcli/scix/workspace/data/sub_neptune_tidallock.remapped.nc \
  --nlat 181 \
  --nlon 360
```

This produces a regular lat-lon file with:

- `time`
- `altitude`
- `lat`
- `lon`

and remapped science fields on dimensions:

```text
(time, altitude, lat, lon)
```

The output follows CF conventions for the coordinate axes and includes
`altitude_bounds` when the input file provides `x1f`.

## 0.5 Degree Example

For a 0.5 degree grid, use 360 latitude cells and 720 longitude cells:

```bash
paddle-cs-remap \
  /home/chengcli/scix/workspace/data/sub_neptune_tidallock.out1.00290.nc \
  /home/chengcli/scix/workspace/data/sub_neptune_tidallock.out2.00290.nc \
  /home/chengcli/scix/workspace/data/sub_neptune_tidallock.remapped_0p5deg.nc \
  --nlat 360 \
  --nlon 720
```

## Selecting Variables

By default, `paddle` remaps all scalar fields it finds and also remaps the
default vector triplet `vel1,vel2,vel3` if those variables exist.

To remap only specific scalar variables:

```bash
paddle-cs-remap \
  in1.nc in2.nc out.nc \
  --nlat 181 \
  --nlon 360 \
  --scalar rho \
  --scalar temp \
  --scalar press
```

To remap a specific vector triplet:

```bash
paddle-cs-remap \
  in1.nc in2.nc out.nc \
  --nlat 181 \
  --nlon 360 \
  --vector vel1,vel2,vel3
```

If the vector triplet is named `vel1,vel2,vel3`, the output variables are:

- `vel_east`
- `vel_north`
- `vel_up`

For other triplet names, the remapper uses `<base>_east`, `<base>_north`, and
`<base>_up`.

## Caching TempestRemap Weights

The first run at a new source-target resolution generates a TempestRemap mesh,
overlap mesh, and offline map. Reuse those files with `--map-cache-dir`:

```bash
paddle-cs-remap \
  in1.nc in2.nc out.nc \
  --nlat 360 \
  --nlon 720 \
  --map-cache-dir /tmp/paddle-tempestremap
```

This avoids rebuilding the map on later runs with the same cubed-sphere
resolution and target grid.

## TempestRemap In A Non-Standard Location

If the TempestRemap binaries are not on `PATH`, point `paddle` at their
directory:

```bash
paddle-cs-remap \
  in1.nc in2.nc out.nc \
  --nlat 181 \
  --nlon 360 \
  --tempest-bin-dir "$HOME/.local/tempestremap/bin"
```

## Metadata Behavior

The remapper preserves source variable metadata when available, including
attributes such as:

- `units`
- `long_name`

Coordinate handling differs slightly from the raw Snapy files:

- the vertical coordinate is renamed from `x1` to `altitude`
- the output uses CF-style axis metadata
- remapped fields are written with `coordinates = "time altitude lat lon"`

## Check The Output

Inspect the resulting file header:

```bash
ncdump -h /home/chengcli/scix/workspace/data/sub_neptune_tidallock.remapped.nc
```

You should see coordinate variables like:

```text
double time(time)
float altitude(altitude)
float altitude_bounds(altitude, bnds)
float lat(lat)
float lon(lon)
```

and data variables on:

```text
(time, altitude, lat, lon)
```

## Notes On Grid Conventions

- The input is a stitched cubed-sphere mosaic, not a lat-lon grid.
- Snapy and TempestRemap do not use the same face ordering internally.
- `paddle` reorders the six faces before applying the TempestRemap map.
- Latitude in the output file is written south-to-north, which matches the
  TempestRemap target mesh ordering used by the current implementation.
