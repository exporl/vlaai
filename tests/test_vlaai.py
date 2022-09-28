"""Tests for the VLAAI network."""
import os
import unittest

import numpy as np
import tensorflow as tf

from scipy.stats import pearsonr

from model import vlaai, pearson_loss, pearson_metric


class VLAAITest(unittest.TestCase):
    """Tests for the pre-trained VLAAI models."""

    def setUp(self) -> None:  # noqa: D102
        self.root_folder = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    def test_model_hdf5(self):
        """Test that the VLAAI model can be loaded from an HDF5 file."""
        model = vlaai()
        model.load_weights(
            os.path.join(self.root_folder, "pretrained_models", "vlaai.h5")
        )
        self._evaluate_on_dtu(model)
        model = tf.keras.models.load_model(
            os.path.join(self.root_folder, "pretrained_models", "vlaai.h5"),
            custom_objects={
                "pearson_loss": pearson_loss,
                "pearson_metric": pearson_metric,
            },
        )
        self._evaluate_on_dtu(model)

    def test_model_savedmodel(self):
        """Test that the VLAAI model can be loaded from a SavedModel."""
        model = tf.keras.models.load_model(
            os.path.join(self.root_folder, "pretrained_models", "vlaai"),
            custom_objects={
                "pearson_loss": pearson_loss,
                "pearson_metric": pearson_metric,
            },
        )
        self._evaluate_on_dtu(model)

    def _evaluate_on_dtu(self, model):
        """Evaluate a VLAAI model on a recording of the DTU dataset.

        Parameters
        ----------
        model: tf.keras.models.Model
            The pre-trained VLAAI model to evaluate.
        """
        data_path = os.path.join(
            self.root_folder, "evaluation_datasets", "DTU", "DTU_S1_000.npz"
        )
        data = np.load(data_path)
        eeg, envelope = data["eeg"], data["envelope"]
        pred = model.predict(tf.expand_dims(eeg, 0))
        reconstruction_score = pearsonr(
            np.squeeze(pred), np.squeeze(envelope)
        )[0]
        self.assertAlmostEqual(
            reconstruction_score, 0.24443195701740353, places=6
        )
