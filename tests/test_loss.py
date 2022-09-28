"""Tests for the loss and metrics functions."""
import unittest

import numpy as np
from scipy.stats import pearsonr

from model import pearson_tf, pearson_loss, pearson_metric


class PearsonTest(unittest.TestCase):
    """Test whether the pearson loss/metrics work."""

    def setUp(self) -> None:  # noqa: D102
        np.random.seed(42)
        self.true = np.random.normal(size=(1, 320, 64))
        self.pred = np.random.normal(size=(1, 320, 64))

    def test_pearson_tf(self):
        """Test the pearson tensorflow function."""
        scores = np.squeeze(pearson_tf(self.true, self.pred).numpy())
        for score_index in range(scores.shape[0]):
            self.assertAlmostEqual(
                scores[score_index],
                pearsonr(
                    self.true[0, :, score_index], self.pred[0, :, score_index]
                )[0],
                places=6,
            )

    def test_pearson_loss(self):
        """Test the pearson loss function."""
        scores = np.squeeze(pearson_loss(self.true, self.pred).numpy())
        for score_index in range(scores.shape[0]):
            self.assertAlmostEqual(
                scores[score_index],
                -pearsonr(
                    self.true[0, :, score_index], self.pred[0, :, score_index]
                )[0],
                places=6,
            )

    def test_pearson_metric(self):
        """Test the pearson metric function."""
        scores = np.squeeze(pearson_metric(self.true, self.pred).numpy())
        for score_index in range(scores.shape[0]):
            self.assertAlmostEqual(
                scores[score_index],
                pearsonr(
                    self.true[0, :, score_index], self.pred[0, :, score_index]
                )[0],
                places=6,
            )
