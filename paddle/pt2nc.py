#!/usr/bin/env python3
"""
Convert a sequence of PyTorch .pt dumps into a CF‐compliant NetCDF4 file
with dimensions (time, x, y, z) plus a 'species' axis for mole fractions.
"""

import os
import tarfile
import re
import torch
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

# ──────────────────────────────── CONFIG ────────────────────────────────
INPUT_DIR = "."
OUTPUT_FILE = "thermo_x_xfrac_to_conc.nc"
# ────────────────────────────────────────────────────────────────────────

# find all .pt files, skip size==0, sort by timestamp
pt_files = []
for fn in os.listdir(INPUT_DIR):
    if not fn.endswith(".pt"):
        continue
    full = os.path.join(INPUT_DIR, fn)
    if os.path.getsize(full) == 0:
        continue
    # expect names like thermo_x_xfrac_to_conc_<epoch>.pt
    m = re.search(r"(\d+)\.pt$", fn)
    if not m:
        continue
    pt_files.append((int(m.group(1)), full))

pt_files.sort(key=lambda x: x[0])
times_epoch = [ts for ts, _ in pt_files]

# load the first file to infer shapes
module = torch.jit.load(pt_files[0][1])
data = {name: param for name, param in module.named_parameters()}

temp0 = data["temp"].numpy()
pres0 = data["pres"].numpy()
xfrac0 = data["xfrac"].numpy()
nx3, nx2, nx1 = temp0.shape
nspecies = xfrac0.shape[3]
nt = len(pt_files)

# pre‐allocate arrays in (time, x1, x2, x3) order
temp_arr = np.empty((nt, nx1, nx2, nx3), dtype=temp0.dtype)
pres_arr = np.empty((nt, nx1, nx2, nx3), dtype=pres0.dtype)
xfrac_arr = np.empty((nspecies, nt, nx1, nx2, nx3), dtype=xfrac0.dtype)

# load all timesteps
for i, (_, path) in enumerate(pt_files):
    module = torch.jit.load(path)
    data = {name: param for name, param in module.named_parameters()}
    t_np = data["temp"].numpy()  # (z, y, x)
    p_np = data["pres"].numpy()  # (z, y, x)
    x_np = data["xfrac"].numpy()  # (species, z, y, x)

    # reorder to (x, y, z)
    temp_arr[i] = t_np.transpose(2, 1, 0)
    pres_arr[i] = p_np.transpose(2, 1, 0)
    for j in range(nspecies):
        xfrac_arr[j, i] = x_np[:, :, :, j].transpose(2, 1, 0)

# create NetCDF4 file
ds = Dataset(OUTPUT_FILE, "w", format="NETCDF4")

# dimensions
ds.createDimension("time", nt)
ds.createDimension("x3", nx3)
ds.createDimension("x2", nx2)
ds.createDimension("x1", nx1)

# coordinate variables
tvar = ds.createVariable("time", "f4", ("time",))
tvar.units = "seconds since 1970-01-01 00:00:00 UTC"
tvar.calendar = "gregorian"
tvar[:] = np.array(times_epoch, dtype="f4")

zvar = ds.createVariable("x1", "f4", ("x1",))
yvar = ds.createVariable("x2", "f4", ("x2",))
xvar = ds.createVariable("x3", "f4", ("x3",))

xvar.axis = "X"
yvar.axis = "Y"
zvar.axis = "Z"

xvar[:] = np.arange(nx3)
yvar[:] = np.arange(nx2)
zvar[:] = np.arange(nx1)

# data variables
temp_v = ds.createVariable("temp", "f4", ("time", "x1", "x2", "x3"), zlib=True)
temp_v.units = "K"
temp_v.long_name = "temperature"

pres_v = ds.createVariable("pres", "f4", ("time", "x1", "x2", "x3"), zlib=True)
pres_v.units = "Pa"
pres_v.long_name = "pressure"

xfrac_v = []
for i in range(nspecies):
    xfrac_v.append(
        ds.createVariable(f"xfrac{i}", "f4", ("time", "x1", "x2", "x3"), zlib=True)
    )
    xfrac_v[i].units = "1"
    xfrac_v[i].long_name = "mole fraction of each species"

# write the data
temp_v[:] = temp_arr
pres_v[:] = pres_arr
for i in range(nspecies):
    xfrac_v[i][:] = xfrac_arr[i]

# global metadata
ds.title = "Debug fields for thermo_x.xfrac_to_conc"
ds.institution = "University of Michigan"
ds.source = "converted from .pt files"
ds.history = f"Created {datetime.utcnow().isoformat()}Z"

ds.close()
print(f"Converted file: {OUTPUT_FILE}")
