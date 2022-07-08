from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [(100, 5000), (100, 10000)],
        'pos_data': [True,False],
    }

    def __init__(self, n_samples=10, n_features=50, pos_data=False, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.pos_data = pos_data
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)
        if self.pos_data:
            X = np.abs(X)
            y = np.abs(y)

        data = dict(X=X, y=y)

        return self.n_features, data
