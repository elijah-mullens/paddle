Paddle technical notes
======================

The first three sections describe radiative forcing through three calculations.
The orbit sets the sunlight incident on each column. Gas and cloud opacity
describe how material absorbs and scatters that light and thermal radiation.
A radiation state combines these inputs, solves vertical transfer, and converts
flux divergence into the energy source used by the dynamical integrator.

Those sections follow that order. They share a single set of
units and shape conventions and culminate in a small six-face Uranus example.
The models are deliberately simple reference implementations; the grey
coefficients and prescribed cloud abundances are demonstration inputs, not a
calibrated Uranus climate model.

The fourth section explains cubed-sphere geometry, restart repartitioning,
and geographic remapping of model output. Further sections can be added
to this collection as new reference implementations are developed.

.. toctree::
   :maxdepth: 2

   orbital/index
   opacity/index
   radiation/index
   cubed_sphere/index

.. only:: html

   Build the combined note
   -----------------------

   To build the combined PDF, install Sphinx, WeasyPrint and tinyhtml5 in your
   Python environment, and MathJax 3 for Node.js. Then run from the repository root:

   .. code-block:: bash

      npm install --prefix /tmp/paddle-pdf-node mathjax-full@3
      python -m sphinx -W -b singlehtml docs/notes /tmp/paddle-radiation-html
      NODE_PATH=/tmp/paddle-pdf-node/node_modules \
          python docs/notes/build_pdf.py \
          --html /tmp/paddle-radiation-html/index.html

   The :download:`PDF builder <build_pdf.py>` uses the accompanying
   :download:`MathJax renderer <render_math.js>` to embed equations as vector
   images before printing. The output is ``docs/notes/paddle_notes.pdf``.
