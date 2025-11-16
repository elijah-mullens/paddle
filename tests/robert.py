import torch
import math
import time
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIPR


def call_user_output(bvars):
    hydro_w = bvars["hydro_w"]
    out = {}
    temp = hydro_w[kIPR] / (Rd * hydro_w[kIDN])
    out["temp"] = temp
    out["theta"] = temp * (p0 / hydro_w[kIPR]).pow(Rd / cp)
    return out


dT = 0.5
p0 = 1.0e5
Ts = 303.15
xc = 500.0
yc = 0.0
zc = 260.0
s = 100.0
a = 50.0
grav = 9.8
Rd = 287.0
gamma = 1.4
uniform_bubble = False

# use cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("robert.yaml")
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.module("hydro.coord")
thermo = block.module("hydro.eos.thermo")

# thermodynamics
cp = gamma / (gamma - 1.0) * Rd

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

temp = Ts - grav * x1v / cp
w[kIPR] = p0 * torch.pow(temp / Ts, cp / Rd)

r = torch.sqrt((x3v - yc) ** 2 + (x2v - xc) ** 2 + (x1v - zc) ** 2)
temp += torch.where(r <= a, dT * torch.pow(w[kIPR] / p0, Rd / cp), 0.0)
if not uniform_bubble:
    temp += torch.where(
        r > a,
        dT * torch.exp(-(((r - a) / s) ** 2)) * torch.pow(w[kIPR] / p0, Rd / cp),
        0.0,
    )
w[kIDN] = w[kIPR] / (Rd * temp)

block_vars = {}
block_vars["hydro_w"] = w
block_vars = block.initialize(block_vars)

block.set_user_output_func(call_user_output)

# integration
start_time = time.time()
current_time = 0.0
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
print("elapsed time = ", time.time() - start_time)
