"""Linear Optimal Low-Rank Projection"""

# Author: Jaewon Chung

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import svd_flip, randomized_svd


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


class LOL(BaseEstimator, TransformerMixin):
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

    svd_solver : string {'auto', 'full', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd`
        randomized :
            run randomized SVD by the method of Halko et al.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'randomized'.

    Attributes
    ----------
    means_ : array-like, shape (n_classes, n_features)
        Class means.
    
    classes_ : array-like, shape (n_classes,)
        Unique class labels.

    priors_ : array-like, shape (n_classes,)
        Class priors (sum to 1).

    n_components_ : int
        Equals the parameter n_components, or n_features if n_components 
        is None.
    """

    def __init__(self,
                 n_components=None,
                 copy=True,
                 svd_solver='auto',
                 random_state=None,
                 iterated_power='auto'):
        self.n_components = n_components
        self.copy = copy
        self.svd_solver = svd_solver
        self.random_state = random_state
        self.iterated_power = iterated_power

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
        """Dispatch to the right submethod depending data shape."""
        X, y = check_X_y(
            X,
            y,
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            copy=self.copy,
            y_numeric=True)

        self.classes_, priors_ = np.unique(y, return_counts=True)
        self.priors_ = priors_ / X.shape[0]

        # Handle n_components==None
        if self.n_components is None:
            n_components = X.shape[1] - len(self.classes_)
        else:
            n_components = self.n_components - len(self.classes_)

        # Handle svd_solver
        svd_solver = self.svd_solver

        if svd_solver == 'auto':
            # Small problem, just call full PCA
            if max(X.shape) <= 500:
                svd_solver = 'full'
            elif n_components >= 1 and n_components < .8 * min(X.shape):
                svd_solver = 'randomized'
            # This is also the case of n_components in (0,1)
            else:
                svd_solver = 'full'

        # Call different fits for either full or truncated SVD
        # Only compute deltas if self.n_components < self.classes_
        if n_components <= 0:
            return self._fit_means(X, y)
        elif svd_solver == 'full':
            return self._fit_full(X, y, n_components)
        elif svd_solver == 'randomized':
            return self._fit_truncated(X, y, n_components)
        else:
            raise ValueError("Unrecognized svd_solver='{0}'"
                             "".format(svd_solver))

    def _fit_means(self, X, y):
        n_samples, n_features = X.shape

        # Compute class means
        self.means_ = _class_means(X, y)

        # Compute difference of means in classes
        delta = self._get_delta(self.means_, self.priors_)

        A = delta.T[:, :self.n_components]

        Q, _ = np.linalg.qr(A)

        self.components_ = Q.T

        return Q

    def _fit_full(self, X, y, n_components):
        """Fit the model by computing full SVD on X"""
        n_samples, n_features = X.shape

        # Compute class means
        self.means_ = _class_means(X, y)

        # Center the data on class means
        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])
        Xc = np.concatenate(Xc, axis=0)

        # Compute difference of means in classes
        delta = self._get_delta(self.means_, self.priors_)

        U, D, V = np.linalg.svd(Xc, full_matrices=False)
        #U, V = svd_flip(U, V, u_based_decision=False)

        # Transpose the V before taking its components
        A = np.concatenate([delta.T, V.T[:, :n_components]], axis=1)

        # Orthognalize and normalize
        Q, _ = np.linalg.qr(A)

        self.components_ = Q.T

        return U, D, V

    def _fit_truncated(self, X, y, n_components):
        """Fir the model by computing truncated SVD on X"""
        random_state = check_random_state(self.random_state)

        n_samples, n_features = X.shape

        if not 1 <= n_components <= n_features:
            raise ValueError("n_components=%r must be between 1 and "
                             "n_features=%r with svd_solver='randomized'" %
                             (n_components, n_features))

        # Get class means
        self.means_ = _class_means(X, y)

        # Center the data on class means
        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])
        Xc = np.concatenate(Xc, axis=0)

        # Compute difference of means in classes
        delta = self._get_delta(self.means_, self.priors_)

        U, D, V = randomized_svd(
            X,
            n_components=n_components,
            n_iter=self.iterated_power,
            flip_sign=True,
            random_state=random_state)

        # Transpose the V before taking its components
        A = np.concatenate([delta.T, V.T[:, :n_components]], axis=1)

        # Orthognalize and normalize
        Q, _ = np.linalg.qr(A)

        self.components_ = Q.T

        return U, D, V

    def _get_delta(self, means, priors):
        """
        Computes the difference of class means in decreasing priors order.
        """
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
        try:
            X_new = X @ self.components_.T
            return X_new

        except AttributeError:
            self._fit(X, y)
            X_new = X @ self.components_.T
            return X_new

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
        >>> from lol import LOL
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> Y = np.array()
        >>> lol = LOL(n_components=2)
        >>> lol.fit(X)
        LOL(copy=True, n_components=2)
        >>> lol.transform(X) # doctest: +SKIP
        """
        check_is_fitted(self, ['components_'], all_or_any=all)

        X = check_array(X)

        X_transformed = np.dot(X, self.components_.T)

        return X_transformed