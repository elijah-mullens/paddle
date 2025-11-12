#! /usr/bin/env python3

"""
Read variables in a NetCDF file and write them to jit saved torch tensors.
Usage: python nc2pt.py input.nc output.pt
"""

from netCDF4 import Dataset
import torch


def save_tensors(tensor_map: dict[str, torch.Tensor], filename: str):
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)  # Needed for LibTorch compatibility
    scripted.save(filename)


fname = "sod.out0.00019.nc"

nc = Dataset(fname, "r")
out_fname = "sod.out0.00019.pt"

data = {}
for varname in nc.variables:
    var = nc.variables[varname][:]
    if var.ndim == 4:  # (time, x1, x2, x3) -> (time, x3, x2, x1)
        data[varname] = torch.tensor(var).permute(0, 3, 2, 1).squeeze()
    elif var.ndim == 3:  # (x1, x2, x3) -> (x3, x2, x1)
        data[varname] = torch.tensor(var).permute(2, 1, 0).squeeze()
    else:
        data[varname] = torch.tensor(var).squeeze()

save_tensors(data, out_fname)
