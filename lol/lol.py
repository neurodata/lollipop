"""Linear Optimal Low-Rank Projection"""

# Author: Jaewon Chung

import numpy as np

from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.base import BaseEstimator


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,)
        Input labels.
    Returns
    -------
    means : array-like, shape (n_features,)
        Class means.
    """
    means = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y == group, :]
        means.append(Xg.mean(0))
    return np.asarray(means)


class LOL(BaseEstimator):
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

    priors_ : array-like, shape (n_classes,)
        Class priors (sum to 1).
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
            X,
            y,
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            copy=self.copy,
            y_numeric=True)

        n_samples, n_features = X.shape
        self.classes_, self.priors_ = np.unique(y, return_counts=True)
        self.priors_ = self.priors_ / n_samples

        # Handle n_components==None
        if self.n_components is None:
            n_components = X.shape[1] - len(self.classes_)
        else:
            n_components = self.n_components - len(self.classes_)

        # Get class means
        self.means_ = _class_means(X, y)

        # Center the data
        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])

        Xc = np.concatenate(Xc, axis=0)

        self.Xc = Xc
        delta = self._get_delta(self.means_, self.priors_)
        #self.delta_ = delta

        U, D, V = np.linalg.svd(Xc, full_matrices=False)
        V = V.T

        A = np.concatenate([delta.T, -V[:, :n_components]], axis=1)

        # Orthognalize and normalize
        Q, _ = np.linalg.qr(A)

        self.components_ = Q.T

    def _get_delta(self, means, priors):
        _, idx = np.unique(priors, return_index=True)
        idx = idx[::-1]
        delta = means.copy()[idx]
        delta[1:] -= delta[0]

        return delta

    def fit_transform(self, X, y):
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
        self._fit(X, y)
        return X @ self.components_.T

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