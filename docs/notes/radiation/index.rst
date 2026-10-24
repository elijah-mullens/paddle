Orbital illumination, opacity, and atmospheric heating
======================================================

Radiative forcing connects three calculations with different responsibilities.
The orbit sets the sunlight incident on each column. Gas and cloud opacity
describe how material absorbs and scatters that light and thermal radiation.
A radiation state combines these inputs, solves vertical transfer, and converts
flux divergence into the energy source used by the dynamical integrator.

The following three sections follow that order. They share a single set of
units and shape conventions and culminate in a small six-face Uranus example.
The models are deliberately simple reference implementations; the grey
coefficients and prescribed cloud abundances are demonstration inputs, not a
calibrated Uranus climate model.

.. toctree::
   :maxdepth: 2

   ../orbital/index
   ../opacity/index
   rt_state
