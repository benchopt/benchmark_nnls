from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from scipy.optimize import _nnls


class Solver(BaseSolver):
    name = 'scipy'

    install_cmd = 'conda'
    requirements = ['scipy>=1.12']

    def set_objective(self, X, y, fit_intercept=False):
        self.X, self.y = X, y
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        self.w, _, _ = _nnls(self.X, self.y, n_iter+1)

    def get_result(self):
        return dict(beta=self.w)
