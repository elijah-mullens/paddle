import torch
import math
import time
import kintera
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIPR


def call_user_output(bvars):
    hydro_w = bvars["hydro_w"]
    out = {}
    temp = hydro_w[kIPR] / (Rd * hydro_w[kIDN])
    out["temp"] = temp
    out["theta"] = temp * (p0 / hydro_w[kIPR]).pow(Rd / cp)
    return out


p0 = 1.0e5
Ts = 300.0
xc = 0.0
xr = 4.0e3
zc = 3.0e3
zr = 2.0e3
dT = -15.0
grav = 9.8
Rd = 287.0
gamma = 1.4
K = 75.0

# use cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("straka.yaml")

# initialize block
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.module("hydro.coord")
thermo = block.module("hydro.eos.thermo")
eos = block.module("hydro.eos")

# thermodynamics
Rd = kintera.constants.Rgas / kintera.species_weights()[0]
cv = kintera.species_cref_R()[0] * Rd
cp = cv + Rd

# setup a meshgrid for simulation
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 5

w = torch.zeros((nvar, nc3, nc2, nc1), device=device)

L = torch.sqrt(((x2v - xc) / xr) ** 2 + ((x1v - zc) / zr) ** 2)
temp = Ts - grav * x1v / cp

w[kIPR] = p0 * torch.pow(temp / Ts, cp / Rd)
temp += torch.where(L <= 1, dT * (torch.cos(L * math.pi) + 1.0) / 2.0, 0)
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
