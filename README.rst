Benchmark repository for Non-negative Least Square
==================================================

|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
The Non-Negative Least Square consists in solving the following program:


$${\\min}_{w \\geq 0} \\frac{1}{2} \\lVert y - Xw \\rVert^2_2$$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and

$$y \\in \\mathbb{R}^n, X = [x_1^\\top, \\dots, x_n^\\top]^\\top \\in \\mathbb{R}^{n \\times p}$$


In case a $w$ with negative entries is passed, those entries are set to 0 to evaluate the objective function at a feasible point.

Install
--------

To download and run the benchmark on a few solvers and datasets, use:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_nnls
   $ benchopt run benchmark_nnls  --config simple_config.yml


Options can be passed to `benchopt run`, e.g. to restrict the benchmarks to some solvers or datasets:

.. code-block::

	$ benchopt run benchmark_nnls -s scipy -d leukemia --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.


.. |Build Status| image:: https://github.com/benchopt/benchmark_nnls/actions/workflows/main.yml/badge.svg
   :target: https://github.com/benchopt/benchmark_nnls/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
