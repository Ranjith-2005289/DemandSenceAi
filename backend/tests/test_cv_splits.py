"""
get_cv_splits used to be a dead conditional in every model file
(`2 if n_obs >= 20 else 2` — both branches always 2), so "best model"
selection was always based on a fixed 2-fold walk-forward evaluation
regardless of how much data was available. These tests pin the corrected,
genuinely adaptive fold-count curve.
"""
import pytest

from services.metrics import get_cv_splits, MIN_CV_SPLITS, MAX_CV_SPLITS


@pytest.mark.parametrize("n_obs,expected", [
    (15, 2),
    (100, 2),
    (199, 2),
    (299, 2),
    (300, 3),
    (399, 3),
    (400, 4),
    (499, 4),
    (500, 5),
    (1000, 5),
    (100000, 5),
])
def test_cv_splits_curve(n_obs, expected):
    assert get_cv_splits(n_obs) == expected


def test_cv_splits_never_below_floor():
    for n_obs in [0, 1, 10, 50]:
        assert get_cv_splits(n_obs) >= MIN_CV_SPLITS


def test_cv_splits_never_above_cap():
    for n_obs in [500, 10_000, 10_000_000]:
        assert get_cv_splits(n_obs) <= MAX_CV_SPLITS
