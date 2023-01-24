.. Evolutionary-Optimization documentation master file, created by
   sphinx-quickstart on Wed Jul 13 15:32:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Evolutionary-Optimization's documentation
=====================================================
This package allows the user to optimise a function using an evolutionary algorithm.
An `evolutionary algorithm <https://en.wikipedia.org/wiki/Evolutionary_algorithm>`_ uses the principles of evolution to find optimal solutions.
At a high level diagram of the process is as follows:

.. mermaid::

   flowchart TD;
       A[Generate population] --> B[Evaluate population];
       B --> C{Is exit condition met?};
       C --No--> D[Update population];
       C --Yes--> E[Exit algorithm];
       D --> B

Currently, the only exit condition for the code is running the desired
number of generations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   quickstart
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
