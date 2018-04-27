import numpy as np
from numpy.testing import assert_almost_equal

from lol.lol import LOL


def test_LOL():
    X = np.random.random((100, 10))
    y = np.random.randint(0, 1, 100)
    l = LOL(n_components=2)
    l.fit(X, y)
    l.transform(X)
    l.fit_transform(X, y)