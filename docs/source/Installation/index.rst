.. role:: bash(code)
   :language: bash

.. _installation:

Installation
=============
To install the Estimate-of-direction-in-RSM repository:

.. code-block:: bash

   $ git clone https://github.com/Megscammell/https://github.com/Megscammell/Estimate-of-direction-in-RSM.git
   $ cd Estimate-direction-in-RSM
   $ python setup.py develop

To ensure all tests are working, create an environment and run the tests using :bash:`pytest`:

.. code-block:: bash

   $ conda env create -f environment.yml
   $ conda activate est_dir_env
   $ pytest
