# -*- coding: utf-8 -*-
# ypo@informatik.uni-kiel.de

import numpy as np
from sklearn.ensemble._gb_losses import (BinomialDeviance, LeastSquaresError,
                                         LossFunction, MultinomialDeviance)

import logger as logging


class ClippedLeastSquaresError(LeastSquaresError):
    """Loss function for clipped least squares (LS) estimation.

    This extension overrides the `LeastSquaresError` method __call__() by
    clipping the squared deviations before summing them.
    It extends `LeastSquaresError`'s constructor by adding the `clipping_bound`
    member.

    Parameters
    ----------
    clipping_bound : float
        The bound used to clip the squared deviations from above and below.
    """

    def __init__(self, clipping_bound):
        super(ClippedLeastSquaresError, self).__init__()
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None):
        """Compute the clipped least squares loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        c = self.clipping_bound
        if sample_weight is None:
            return np.mean(np.clip((y - raw_predictions.ravel()) ** 2, -c, c))
        else:
            raise NotImplementedError(
                "Clipping is not implemented if argument `sample_weight` is "
                "not None."
            )
            return (1 / sample_weight.sum() * np.sum(
                sample_weight * ((y - raw_predictions.ravel()) ** 2)))

class ClippedBinomialDeviance(BinomialDeviance):
    def __init__(self, n_classes, clipping_bound):
        super(ClippedBinomialDeviance, self).__init__(n_classes = n_classes)
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None):
        raise NotImplementedError()

class ClippedMultinomialDeviance(MultinomialDeviance):
    def __init__(self, n_classes, clipping_bound):
        super(ClippedMultinomialDeviance, self).__init__(n_classes = n_classes)
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None):
        raise NotImplementedError()

LOSS_FUNCTIONS = {
    'ls'       : LeastSquaresError,
    'c_ls'      : ClippedLeastSquaresError,
    'bin_dev'  : BinomialDeviance,
    'c_bin_dev' : ClippedBinomialDeviance,
    'mul_dev'  : MultinomialDeviance,
    'c_mul_dev' : ClippedMultinomialDeviance
}
