import numpy as np

class LossPois:
    def __init__(self, n_obs=1000,  n_relevant=300, seed=None):

        assert n_obs >= n_relevant

        if seed is not None:
            np.random.seed(seed)

        X = np.random.uniform(0, 0.1, (n_obs - n_relevant, 2))
        self.X_ = np.concatenate([X, np.random.uniform(0, 2, (n_relevant, 2))], axis=0)
        self.weights_ = np.ones(n_obs)
        self.update_numbers_ = np.ones(n_obs)

    def __call__(self, w, sample_share=None, update_weights=False):
        if sample_share == None:
            energy = np.abs(self.X_.dot(w))
            positive_mask = np.random.poisson(lam=energy, size=len(energy)) > 0
            # print(np.sum(positive_mask[:700]))
            # print(np.sum(positive_mask[700:]))
            return np.sum(energy[positive_mask])

        # sample from the weighted distribution
        proba_weights = self.weights_ / self.update_numbers_ / np.sum(self.weights_ / self.update_numbers_)
        sample_mask = np.random.choice(np.arange(len(self.X_)),
                                       int(sample_share * len(self.X_)),
                                       p=proba_weights,
                                       replace=True
                                      )

        # compute the values
        energy = np.abs(self.X_[sample_mask].dot(w))
        positive_mask = np.random.poisson(lam=energy, size=len(energy)) > 0

        # update weights
        if update_weights is True:
            self.weights_[sample_mask] += energy * positive_mask
            self.update_numbers_[sample_mask] += 1

        # renormalize weights for unbiased estimation
        return np.sum(energy[positive_mask] / proba_weights[sample_mask][positive_mask]) / len(energy)


import numpy as np

class LossUniform:
    def __init__(self, n_obs=1000,  n_relevant=300, seed=None):

        assert n_obs >= n_relevant

        if seed is not None:
            np.random.seed(seed)

        X = np.random.uniform(0, 0.1, (n_obs - n_relevant, 2))
        self.X_ = np.concatenate([X, np.random.uniform(0, 2, (n_relevant, 2))], axis=0)
        self.bias_ = np.random.uniform(0, 1, n_obs)
        self.weights_ = np.ones(n_obs)
        self.update_numbers_ = np.ones(n_obs)

    def __call__(self, w, sample_share=None, update_weights=False):
        if sample_share == None:
            energy = np.abs(self.X_.dot(w) + self.bias_)
            positive_mask = np.random.uniform(0, 1, size=len(energy)) < energy
            # print(np.sum(positive_mask[:700]))
            # print(np.sum(positive_mask[700:]))
            return np.sum(energy[positive_mask])

        # sample from the weighted distribution
        proba_weights = self.weights_ / self.update_numbers_ / np.sum(self.weights_ / self.update_numbers_)
        sample_mask = np.random.choice(np.arange(len(self.X_)),
                                       int(sample_share * len(self.X_)),
                                       p=proba_weights,
                                       replace=True
                                      )

        # compute the values
        energy = np.abs(self.X_[sample_mask].dot(w) + self.bias_[sample_mask])
        positive_mask = np.random.uniform(0, 1, size=len(energy)) < energy

        # update weights
        if update_weights is True:
            self.weights_[sample_mask] += energy * positive_mask
            self.update_numbers_[sample_mask] += 1

        # renormalize weights for unbiased estimation
        return np.sum(energy[positive_mask] / proba_weights[sample_mask][positive_mask]) / len(energy)


class LossAngry:
    def __init__(self, n_obs=1000,  n_relevant=300, seed=None):

        assert n_obs >= n_relevant

        if seed is not None:
            np.random.seed(seed)

        X = np.random.uniform(0, 0.1, (n_obs - n_relevant, 2))
        self.X_ = np.concatenate([X, np.random.uniform(0, 2, (n_relevant, 2))], axis=0)
        self.weights_ = np.ones(n_obs)
        self.update_numbers_ = np.ones(n_obs)
        self.bias_ = np.random.uniform(-1, 1, n_obs)

    def __call__(self, w, sample_share=None, update_weights=False):
        if sample_share == None:
            energy = np.abs(np.tan(self.X_.dot(w) + self.bias_))
            positive_mask = np.random.poisson(lam=energy, size=len(energy)) > 0
            # print(np.sum(positive_mask[:700]))
            # print(np.sum(positive_mask[700:]))
            return np.sum(energy[positive_mask])

        # sample from the weighted distribution
        proba_weights = self.weights_ / self.update_numbers_ / np.sum(self.weights_ / self.update_numbers_)
        sample_mask = np.random.choice(np.arange(len(self.X_)),
                                       int(sample_share * len(self.X_)),
                                       p=proba_weights,
                                       replace=True
                                      )

        # compute the values
        energy = np.abs(np.tan(self.X_[sample_mask].dot(w) + self.bias_[sample_mask]))
        positive_mask = np.random.poisson(lam=energy, size=len(energy)) > 0

        # update weights
        if update_weights is True:
            self.weights_[sample_mask] += energy * positive_mask
            self.update_numbers_[sample_mask] += 1

        # renormalize weights for unbiased estimation
        return np.sum(energy[positive_mask] / proba_weights[sample_mask][positive_mask]) / len(energy)
