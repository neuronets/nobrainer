"""Checkpointing utils"""

import glob
import os
import tensorflow as tf

from .base import BaseEstimator


class CheckpointTracker(tf.keras.callbacks.Callback):
    """Class for saving/loading estimators at/from checkpoints."""

    def __init__(self, processor, checkpoint_dir):
        self.processor = processor
        self.checkpoint_dir = checkpoint_dir
        self._checkpoint_stem = os.path.join(checkpoint_dir, 'checkpoint-epoch')

    def load(self):
        latest = max(glob.glob(self._checkpoint_stem + '*'),
                     key=os.path.getctime)
        self.processor = self.processor.load(latest)
        return self.processor

    def save(self, suffix):
        save_dir = self._checkpoint_stem + suffix
        print(f"Saving to dir {save_dir}")
        self.processor.save(save_dir)

    def on_epoch_end(self, epoch, logs=None):
        print(f"End epoch {epoch}, saving")
        self.save(f'_{epoch:03d}')
