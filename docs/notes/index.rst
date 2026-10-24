Technical notes
===============

These notes connect Paddle's reference implementations to their equations,
coordinate conventions, and reproducible numerical examples. Each topic has
its own directory containing an ``index.rst``, figure-generation scripts, and
included figures. Future notes can be added to this contents list.

.. toctree::
   :maxdepth: 2

   radiation/index

Build the collection with Sphinx installed, from the repository root:

.. code-block:: bash

   python -m sphinx -W -b html docs/notes /tmp/paddle-notes-html
