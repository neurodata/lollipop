"""Linear Optimal Low-Rank Projection"""

# Author: Jaewon Chung

import numpy as np

from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


class LOL():
    """
    Linear Optimal Low-Rank Projection (LOL)

    Supervised linear dimensionality reduction using Singular Value 
    Decomposition of the data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    Attributes
    ----------
    means_ : array-like, shape (n_classes, n_features)
        Class means.
    
    classes_ : array-like, shape (n_classes,)
        Unique class labels.
    """

    def __init__(self,
                 n_components=None,
                 copy=True,
                 tol=0.0,
                 random_state=None):

        self.n_components = n_components
        self.copy = copy
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model with X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples, )
            Labels for training data, where n_samples is the number
            of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        X, y = check_X_y(
            X, y, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy,
            y_numeric=True)

        n_samples, n_features = X.shape
        self.classes_, self.priors_ = np.unique(y, return_counts=True)
        self.priors_ /= n_samples

        # Handle n_components==None
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # Center the data
        self.mean_ = np.empty((len(classes), n_features)
        for idx, k in enumerate(classes):
            rdx = y == k
            mu = np.mean(X[rdx], axis=0)
            X[rdx] -= mu
            self.mean_[idx] = mu

        delta = _get_delta(self.mean_, self.priors_)

    def _get_delta(self, mean, priors):
        _, idx = np.unique(priors, return_index=True)
        delta = mean.copy()
        delta[1:] -= delta[0]

        return delta

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples, )
            Labels for training data, where n_samples is the number
            of samples.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        pass

    def transform(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.decomposition import IncrementalPCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> ipca = IncrementalPCA(n_components=2, batch_size=3)
        >>> ipca.fit(X)
        IncrementalPCA(batch_size=3, copy=True, n_components=2, whiten=False)
        >>> ipca.transform(X) # doctest: +SKIP
        """
        check_is_fitted(self, ['mean_', 'components_'], all_or_any=all)

        X = check_array(X)

        if self.mean_ is not None:
            X = X - self.mean_

        X_transformed = np.dot(X, self.components_.T)

        return X_transformed