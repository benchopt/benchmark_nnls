from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from skglm.penalties import PositiveConstraint
    from skglm.datafits import Quadratic
    from skglm.solvers import AndersonCD
    from skglm.estimators import GeneralizedLinearEstimator


class Solver(BaseSolver):
    name = 'skglm'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/skglm.git@main'
    ]

    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel '
        'and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    def set_objective(self, X, y, fit_intercept=False):
        self.X, self.y = np.asfortranarray(X), y
        self.fit_intercept = fit_intercept

        self.clf = GeneralizedLinearEstimator(
            datafit=Quadratic(),
            penalty=PositiveConstraint(),
            solver=AndersonCD(tol=1e-9, fit_intercept=fit_intercept)
        ).fit(X, y)

    def run(self, n_iter):
        self.clf.solver.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return dict(beta=self.clf.coef_.flatten())
