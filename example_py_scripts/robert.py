import torch
import math
import kintera
import os
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
uniform_bubble = False

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("robert.yaml")
block = MeshBlock(op)

# use cuda if available
if torch.cuda.is_available() and op.layout().backend() == "nccl":
    device = torch.device(block.device())
else:
    device = torch.device("cpu")

block.to(device)
block.set_user_output_func(call_user_output)

# get handles to modules
coord = block.module("coord")
eos = block.module("hydro.eos")
grav = -block.options.hydro().grav().grav1()

# thermodynamics
gamma = eos.options.gammad()
Rd = kintera.constants.Rgas / eos.options.weight()
cp = gamma / (gamma - 1.0) * Rd

# set initial condition
x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)

# initialize from "robert.final.restart" if exists
if os.path.exists("robert.final.restart"):
    block_vars, current_time = block.initialize_from_restart("robert.final.restart")
    # extend the time by a factor of 2
    block.options.intg().tlim(2.0 * block.options.intg().tlim())
    print(
        "Initialized from robert.final.restart, extending time to",
        block.options.intg().tlim(),
    )
else:
    # dimensions
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]

    w = torch.zeros((eos.nvar(), nc3, nc2, nc1), device=device)

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

    block_vars, current_time = block.initialize({"hydro_w": w})

# integration
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
