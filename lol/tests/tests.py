import numpy as np
# from numpy.testing import assert_almost_equal

from lol.lol import LOL


def test_LOL():
    X = np.random.random((100, 10))
    y = np.random.randint(0, 2, 100)
    lmao = LOL(n_components=3)
    lmao.fit(X, y)
    lmao.transform(X)
    lmao.fit_transform(X, y)
