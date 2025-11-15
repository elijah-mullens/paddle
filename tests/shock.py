import torch
from snapy import index, MeshBlockOptions, MeshBlock

torch.set_default_dtype(torch.float64)

# device
device = torch.device("cuda:0")

# set hydrodynamic model
op = MeshBlockOptions.from_yaml("shock.yaml")

# initialize block
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.hydro.module("coord")

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 5

w = torch.zeros((nvar, nc3, nc2, nc1), device=device)

w[index.idn] = torch.where(x1v < 0.0, 1.0, 0.125)
w[index.ipr] = torch.where(x1v < 0.0, 1.0, 0.1)
w[index.ivx] = w[index.ivy] = w[index.ivz] = 0.0

# internal boundary
r1 = torch.sqrt(x1v * x1v + x2v * x2v + x3v * x3v)
solid = torch.where(r1 < 0.1, 1, 0).to(torch.bool)

block_vars = {}
block_vars["hydro_w"] = w
block_vars["solid"] = solid
block_vars = block.initialize(block_vars)

# integration
current_time = 0.0
block.make_outputs(block_vars, current_time)

while not block.intg.stop(block.inc_cycle(), current_time):
    dt = block.max_time_step(block_vars)
    block.print_cycle_info(current_time, dt)

    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, block_vars)

    current_time += dt
    block.make_outputs(block_vars, current_time)
