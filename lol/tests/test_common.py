from sklearn.utils.estimator_checks import check_estimator
from lol import LOL


def test_transformer():
    return check_estimator(LOL)