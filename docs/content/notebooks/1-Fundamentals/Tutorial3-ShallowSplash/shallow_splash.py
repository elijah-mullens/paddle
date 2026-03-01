import torch
import time
import numpy as np
from snapy.distributed import get_rank, get_layout
from snapy.coord import get_cs_face_name, cs_ab_to_lonlat
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIV2, kIV3

phi = 500.0
dphi = 10.0
radius = 5.0e5

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("shallow_splash.yaml", verbose=False)
op.output_dir("/data")
block = MeshBlock(op)

# use cuda if available
if torch.cuda.is_available() and op.layout().backend() == "nccl":
    device = torch.device(block.device())
else:
    device = torch.device("cpu")

block.to(device)

# get handles to modules
coord = block.module("coord")

# set coordinates
r = get_rank()

layout = get_layout()
rx, ry, face_id = layout.loc_of(r)
face = get_cs_face_name(face_id)

beta, alpha, r_planet = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)
lon, lat = cs_ab_to_lonlat(face, alpha, beta)

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 4

w = torch.zeros((nvar, nc3, nc2, nc1), device=device)

dist = r_planet * (np.pi / 2.0 - lat)

w[kIDN] = phi
w[kIDN][torch.logical_and(dist < radius, lat > np.pi / 4.0)] += dphi
w[kIV2] = 0.0
w[kIV3] = 0.0

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
