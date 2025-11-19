from scipy.interpolate import interp1d
import snapy
import numpy as np

from .setup_profile import setup_profile


def find_init_params(
    block: snapy.MeshBlock,
    param: dict[str, float],
    *,
    target_T: float = 300.0,
    target_P: float = 1.0e5,
    method: str = "moist-adiabat",
    max_iter: int = 50,
    ftol: float = 1.0e-2,
    verbose: bool = True,
):
    """Find initial parameters that yield desired T and P

    Args:
        block (snapy.MeshBlock): The mesh block to set up.
        param (dict[str, float]): Initial guess parameters for the adiabat setup.
            Required keys: Ts, Ps, x<species>, grav.
        target_T (float, optional): Target temperature in Kelvin. Defaults to 300 K.
        target_P (float, optional): Target pressure in Pascals. Defaults to 1e5 Pa.
        method (str, optional): Method for the adiabat setup.
        max_iter (int, optional): Maximum number of iterations. Defaults to 50.
        ftol (float, optional): Tolerance for temperature convergence. Defaults to 1e-2 K.
        verbose (bool, optional): If True, print iteration details. Defaults to True.

    Returns:
        dict: Dictionary containing the found parameters: Ts, Ps, xH2O, xNH3, xH2S.
    """
    count = 0
    eos = block.hydro.get_eos()

    while count < max_iter:
        if verbose:
            print(f"Iteration {count+1}: Trying Ts={param['Ts']}\n")

        # setup profile
        w = setup_profile(block, param, method=method)

        # calculate temperature
        temp = eos.compute("W->T", (w,)).squeeze()

        # calculate 1D pressure
        pres = w[snapy.index.ipr, ...].squeeze()

        # temperature function
        t_func = interp1d(
            pres.log().cpu().numpy(),
            temp.log().cpu().numpy(),
            kind="linear",
            fill_value="extrapolate",
        )

        temp1 = np.exp(t_func(np.log(target_P)))
        if verbose:
            print(f"  At P={target_P:.3e} Pa, T={temp1:.3f} K (target {target_T} K)\n")
        if abs(temp1 - target_T) < ftol:
            if verbose:
                print("Converged! Found parameters:")
                for key, val in param.items():
                    print(f"  {key} = {val}")
                print(f"Matching T = {target_T} K at P = {target_P} Pa")
            return param

        # adjust Ts using a damped scaling
        param["Ts"] += (target_T - temp1) * 0.9
        count += 1

    raise RuntimeError("Failed to converge within the maximum number of iterations.")
