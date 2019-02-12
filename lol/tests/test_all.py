import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from lol.lol import LOL


def generate_data(n_samples=100, n_features=5, n_classes=2):
    X = np.random.random((n_samples, n_features))
    y = np.random.randint(0, n_classes, n_samples)

    return X, y


def test_LOL():
    # Data parameters
    n_samples = 100
    n_features = 5
    n_classes = 2

    X, y = generate_data(n_samples, n_features, n_classes)

    lmao = LOL(n_components=3)
    lmao.fit(X, y)
    lmao.transform(X)
    lmao.fit_transform(X, y)


def test_fit_transform():
    """Test the transform and fit_transform method. Ensure
    outputs are equal
    """
    X, y = generate_data()

    lmao = LOL()
    lmao.fit(X, y)
    X_lmao = lmao.transform(X)

    rofl = LOL()
    X_rofl = rofl.fit_transform(X, y)

    assert_equal(X_lmao, X_rofl)


def test_delta():
    """Test different deltas. Ensure number of vectors you recieve is
    n_classes - 1
    """
    for n_classes in range(2, 5):
        X, y = generate_data(n_classes=n_classes)
        lmao = LOL(n_components=n_classes + 1, svd_solver='randomized')
        lmao.fit(X, y)
        shape = lmao.delta_.shape

        assert_equal(shape[0], n_classes - 1)


def test_orthgonalize():
    """Ensure components have unit length
    """
    n_features = 5
    X, y = generate_data(n_features=n_features)

    for n_components in range(1, n_features + 1):
        lmao = LOL(n_components=n_components, orthogonalize=True)
        lmao.fit(X, y)

        components = np.linalg.norm(lmao.components_, axis=1)
        target = np.ones(len(components))

        assert_almost_equal(components, target)
