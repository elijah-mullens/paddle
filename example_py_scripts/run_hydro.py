import torch
import math
import time
import numpy as np
import yaml
import snapy
import kintera
from snapy import MeshBlockOptions, MeshBlock, OutputOptions, NetcdfOutput
from kintera import ThermoX, KineticsOptions, Kinetics, evolve_implicit
from paddle import (
    setup_profile,
    evolve_kinetics,
)

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    infile = "jupiter_gcm.yaml"
    config = yaml.safe_load(open(infile, "r"))

    # use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(op)
    block.to(device)

    # get handles to modules
    coord = block.module("hydro.coord")
    thermo_y = block.module("hydro.eos.thermo")
    eos = block.hydro.get_eos()
    # thermo_y.options.max_iter(100)

    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    param = {}
    param["Ts"] = float(config["problem"]["Ts"])
    param["Ps"] = float(config["problem"]["Ps"])
    param["grav"] = -float(config["forcing"]["const-gravity"]["grav1"])
    param["Tmin"] = float(config["problem"]["Tmin"])
    for name in thermo_y.options.species():
        param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))

    block_vars = {}
    block_vars["hydro_w"] = setup_profile(block, param, method="pseudo-adiabat")
    block_vars, current_time = block.initialize(block_vars)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # integration
    start_time = time.time()
    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(count, current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)

        err = block.check_redo(block_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        # evolve_kinetics(block_vars, eos, thermo_x, thermo_y, kinet, dt)

        current_time += dt
        block.make_outputs(block_vars, current_time)

block.finalize(block_vars, current_time)
