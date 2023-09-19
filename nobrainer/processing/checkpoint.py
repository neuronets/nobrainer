"""Checkpointing utils"""

from glob import glob
import logging
import os

import tensorflow as tf


class CheckpointTracker(tf.keras.callbacks.ModelCheckpoint):
    """Class for saving/loading estimators at/from checkpoints."""

    def __init__(self, estimator, file_path, **kwargs):
        """
        estimator: BaseEstimator, instance of an estimator (e.g., Segmentation).
        file_path: str, directory to/from which to save or load.
        """
        self.estimator = estimator
        super().__init__(file_path, **kwargs)

    def _save_model(self, epoch, batch, logs):
        """Save the current state of the estimator. This overrides the
        base class implementation to save `nobrainer` specific info.

        epoch: int, the index of the epoch that just finished.
        batch: int, the index of the batch that just finished.
        logs: dict, logging info passed into on_epoch_end or on_batch_end.
        """
        self.save(self._get_file_path(epoch, batch, logs))

    def save(self, directory):
        """Save the current state of the estimator.
        directory: str, path in which to save the model.
        """
        logging.info(f"Saving to dir {directory}")
        self.estimator.save(directory)

    def load(
        self,
        multi_gpu=True,
        custom_objects=None,
        compile=False,
        model_args=None,
    ):
        """Loads the most-recently created checkpoint from the
        checkpoint directory.
        """
        checkpoints = glob(os.path.join(os.path.dirname(self.filepath), "*/"))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=os.path.getctime)
        self.estimator = self.estimator.load(
            latest,
            multi_gpu=multi_gpu,
            custom_objects=custom_objects,
            compile=compile,
        )
        logging.info(f"Loaded estimator from {latest}.")
        return self.estimator
