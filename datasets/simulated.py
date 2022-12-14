from benchopt import BaseDataset, safe_import_context
from benchopt.datasets.simulated import make_correlated_data


with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [(100, 5000), (100, 10000)],
        'positive': [True, False],
    }

    def __init__(self, n_samples=10, n_features=50, positive=False, rho=0,
                 snr=3, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.positive = positive
        self.random_state = random_state
        self.rho = rho
        self.snr = snr

    def get_data(self):

        rng = np.random.RandomState(self.random_state)

        X, y, w_true = make_correlated_data(self.n_samples, self.n_features,
                                            rho=self.rho, random_state=rng)

        if self.positive:
            X = np.abs(X)
            w_true = np.abs(w_true)
            # Regenerate y
            y = X @ w_true
            noise = rng.randn(self.n_samples)
            if self.snr not in [0, np.inf]:
                y += noise / norm(noise) * norm(y) / self.snr
            elif self.snr == 0:
                y = noise

        data = dict(X=X, y=y)

        return data
