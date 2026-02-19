import torch

import time
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIV2, kIV3

phi = 10.0
uphi = 10.0
dphi = 2.0

# use cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("shallow_yz.yaml", verbose=False)

# initialize block
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.module("coord")

# set initial condition
x3v, x2v, _ = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 4

w = torch.zeros((nvar, nc3, nc2, nc1), device=device)

w[kIDN] = phi
w[kIDN][torch.logical_and(x3v > 0.0, x3v < 5.0)] += dphi
w[kIV3] = torch.where(x2v > 0.0, -uphi / w[kIDN], uphi / w[kIDN])
w[kIV2] = 0.0

block_vars = {}
block_vars["hydro_w"] = w
block_vars, current_time = block.initialize(block_vars)

# integration
start_time = time.time()
block.make_outputs(block_vars, current_time)

while not block.intg.stop(block.inc_cycle(), current_time):
    dt = block.max_time_step(block_vars)
    block.print_cycle_info(block_vars, current_time, dt)

    for stage in range(len(block.intg.stages)):
        block.forward(block_vars, dt, stage)

    err = block.check_redo(block_vars)
    if err > 0:
        continue  # redo current step
    if err < 0:
        break  # terminate

    current_time += dt
    block.make_outputs(block_vars, current_time)

block.finalize(block_vars, current_time)
