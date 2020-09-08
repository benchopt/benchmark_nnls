Benchmark repository for Non-negative Least Square
==================================================

|Build Status| |Python 3.6+| |codecov|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

	$ pip install -U benchopt
        $ git clone https://github.com/benchopt/benchmark_nnls
        $ benchopt run benchmark_nnls

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmarks_nnls -s sklearn -d boston --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.


List of optimization problems available
---------------------------------------

Notation:  In what follows, n (or n_samples) stands for the number of samples and p (or n_features) stands for the number of features.

.. math::

 y \in \mathbb{R}^n, X = [x_1^\top, \dots, x_n^\top]^\top \in \mathbb{R}^{n \times p}

- `nnls`: non-negative least-squares. This consists in solving the following program:

.. math::

    \min_{w \geq 0} \frac{1}{2} \|y - Xw\|^2_2


.. |Build Status| image:: https://dev.azure.com/benchopt/benchopt/_apis/build/status/benchopt.benchOpt?branchName=master
   :target: https://dev.azure.com/benchopt/benchopt/_build/latest?definitionId=1&branchName=master
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
.. |codecov| image:: https://codecov.io/gh/benchopt/benchOpt/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/benchopt/benchmark_nnls
