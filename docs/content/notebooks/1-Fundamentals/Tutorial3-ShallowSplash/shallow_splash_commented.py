# Parse command-line arguments passed to the script
import argparse

# Interact with the operating system (set environment variables, create directories)
import os

# Perform optimized numerical array operations
import numpy as np

# Primary deep learning and tensor computation library
import torch

# Read and parse YAML configuration files
import yaml

# Import structures to manage the simulation mesh grid and its settings
from snapy import Mesh, MeshOptions

# Import specific indexes for accessing variables (Density, Velocity Y, Velocity Z)
from snapy import kIDN, kIV2, kIV3

# Import helper functions to convert grid points to Longitude/Latitude and find grid face names
from snapy.coord import cs_ab_to_lonlat, get_cs_face_name


# Defines a function to set up physical values (like fluid depth or height) inside a sub-grid block
def initialize_block(
    block, config: dict, device: torch.device
) -> dict[str, torch.Tensor]:
    # Extract baseline height/fluid value from the YAML configuration
    phi = float(config["problem"]["phi"])
    # Extract the splash disturbance delta height value from the YAML configuration
    dphi = float(config["problem"]["dphi"])
    # Extract the physical radius size of the splash disturbance area
    radius = float(config["problem"]["radius"])

    # Fetch the coordinate module properties from this specific sub-grid block
    coord = block.module("coord")
    # Fetch the spatial structural layout metadata of this sub-grid block
    layout = block.get_layout()
    # Extract the internal coordinate face ID number mapped to this parallel worker process
    _, _, face_id = layout.loc_of(layout.options.rank())
    # Convert that face ID number into a readable human name (e.g., top, bottom, front)
    face = get_cs_face_name(face_id)

    # Generate a 3D coordinate grid matrix (beta=X3, alpha=X2, r_planet=X1) across the sub-block space
    beta, alpha, r_planet = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )
    # Convert the raw grid cube coordinates into true spherical longitude and latitude arrays
    _, lat = cs_ab_to_lonlat(face, alpha, beta)

    # Read total number of grid cells along the 3rd axis
    nc3 = coord.buffer("x3v").shape[0]
    # Read total number of grid cells along the 2nd axis
    nc2 = coord.buffer("x2v").shape[0]
    # Read total number of grid cells along the 1st axis
    nc1 = coord.buffer("x1v").shape[0]
    # Declare the number of simulation variables per cell (Density, 3 Velocities)
    nvar = 4

    # Create an empty, zeroed-out 4D tensor matrix matching our grid dimensions on the active device
    w = torch.zeros((nvar, nc3, nc2, nc1), device=device)
    # Calculate the great-circle surface distance from the designated splash zone center point
    gc_dist = r_planet * (np.pi / 2.0 - lat)

    # Populate the primary density/fluid height index with the baseline 'phi' value
    w[kIDN] = phi
    # Apply a conditional mask: add extra 'dphi' height where cells fall inside the splash radius and northern hemisphere
    w[kIDN][torch.logical_and(gc_dist < radius, lat > np.pi / 4.0)] += dphi
    # Initialize the secondary velocity directional variables to 0.0 (perfectly stationary fluid)
    w[kIV2] = 0.0
    w[kIV3] = 0.0

    # Wrap the created state array into a dictionary labeled for the hydrodynamic solver
    return {"hydro_w": w}


# Entry point function containing the execution flow of the simulation
def main() -> None:
    # Instantiate the CLI parsing interface tool
    parser = argparse.ArgumentParser()
    # Create parameter option pointing to the configuration file path
    parser.add_argument("--input", default="shallow_splash_made_in_notebook.yaml")
    # Create parameter option specifying where simulation results should save
    parser.add_argument("--output-dir", default="./splash_results")
    # Read parameters entered by the user at launch time
    args = parser.parse_args()

    # Safely generate the output directory paths on disk if they do not yet exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Open the assigned configuration file with explicit safe text reading modes
    with open(args.input, "r", encoding="utf-8") as stream:
        # Load the structured configuration dictionary into memory
        config = yaml.safe_load(stream)

    # Convert the raw input configuration file directly into a verified MeshOptions object instance
    options = MeshOptions.from_yaml(args.input, verbose=False)
    # Use the device selected by the DEVICE and DEVICE_ID environment variables
    device = torch.device(options.device_str())
    # Configure each internal block to output its tracking logs directly to the correct output folder
    options.block().output_dir(args.output_dir)

    # Allocate and generate the actual underlying mathematical simulation grid object
    mesh = Mesh(options)
    # Push the entire grid matrix and its properties onto the target CPU or GPU device memory
    mesh.to(device)

    # Execute initialization code for every single local block present inside this mesh slice
    block_vars = [initialize_block(block, config, device) for block in mesh.blocks]
    # Set up memory buffers across the full grid and extract the system starting simulation time
    block_vars, current_time = mesh.initialize(block_vars)
    # Export the initial condition snapshot to disk before stepping forward
    mesh.make_outputs(block_vars, current_time)

    # Fetch the simulation time integrator module belonging to the first mesh sub-block
    intg = mesh.module("block0.intg")
    # Initialize the frame/step cycle tracking counter to zero
    cycle = 0
    # Keep running the physics engine loop until the integrator flags that time/cycle constraints are reached
    while not intg.stop(cycle, current_time):
        # Step up the cycle count tracking integer
        cycle += 1
        # Sync the core mesh module state tracker to match this new cycle ID step
        mesh.set_cycle(cycle)

        # Compute the maximum safe delta time step (dt) to keep the simulation stable without crashing
        dt = mesh.max_time_step(block_vars)
        # Log active step metrics (current time, delta time, cycle) straight to terminal output
        mesh.print_cycle_info(block_vars, current_time, dt)

        # Loop through every mathematical sub-stage required by this step integrator (e.g., Runge-Kutta stages)
        for stage in range(len(intg.stages)):
            # Update values forward through time across this individual numeric integration phase
            mesh.forward(block_vars, dt, stage)

        # Verify numerical values to make sure variables did not spiral out of safe bounds
        err = mesh.check_redo(block_vars)
        # Condition check: if error code indicates step instability, repeat calculation step with shorter time step
        if err > 0:
            continue
        # Condition check: if error code indicates terminal failure, immediately break and halt the script
        if err < 0:
            break

        # Increment simulation clock forward by the safe computed delta time duration
        current_time += dt
        # Save updated physical metrics data arrays out to output file trackers on disk
        mesh.make_outputs(block_vars, current_time)

    # Clean up simulation allocations and finalize open telemetry files upon hitting target runtime boundaries
    mesh.finalize(block_vars, current_time)


# Safeguard to ensure this code executes only when launched directly (and not when imported elsewhere)
if __name__ == "__main__":
    main()
