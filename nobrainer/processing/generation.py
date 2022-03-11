# import importlib
import os
from pathlib import Path

import tensorflow as tf

from .base import BaseEstimator
from .. import losses
from ..dataset import get_dataset


class ProgressiveGeneration(BaseEstimator):
    """Perform generation type operations"""

    state_variables = ["current_resolution_"]

    def __init__(
        self,
        latent_size=256,
        label_size=0,
        num_channels=1,
        dimensionality=3,
        g_fmap_base=1024,
        d_fmap_base=1024,
    ):
        self.model_ = None
        self.latent_size = latent_size
        self.label_size = label_size
        self.g_fmap_base = g_fmap_base
        self.d_fmap_base = d_fmap_base
        self.num_channels = num_channels
        self.dimensionality = dimensionality
        self.current_resolution_ = 0

    def fit(
        self,
        dataset_train,
        epochs=2,
        checkpoint_dir=Path(os.getcwd()) / "temp",
        # TODO: figure out whether optimizer args should be flattened
        g_optimizer=None,
        g_opt_args=None,
        d_optimizer=None,
        d_opt_args=None,
        g_loss=losses.wasserstein,
        d_loss=losses.wasserstein,
        warm_start=False,
        multi_gpu=False,
        num_parallel_calls=None,
    ):
        """Train a progressive gan model"""
        # TODO: check validity of datasets

        # create checkpoint sub-dirs
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        generated_dir = checkpoint_dir / "generated"
        model_dir = checkpoint_dir / "saved_models"
        log_dir = checkpoint_dir / "logs"

        generated_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

        # set optimizers
        g_opt_args = g_opt_args or {}
        if g_optimizer is None:
            g_optimizer = tf.keras.optimizers.Adam
        g_opt_args_tmp = dict(
            learning_rate=1e-04, beta_1=0.0, beta_2=0.99, epsilon=1e-8
        )
        g_opt_args_tmp.update(**g_opt_args)
        g_opt_args = g_opt_args_tmp

        d_opt_args = d_opt_args or {}
        if d_optimizer is None:
            d_optimizer = tf.keras.optimizers.Adam
        d_opt_args_tmp = dict(
            learning_rate=1e-04, beta_1=0.0, beta_2=0.99, epsilon=1e-8
        )
        d_opt_args_tmp.update(**d_opt_args)
        d_opt_args = d_opt_args_tmp

        if warm_start:
            if self.model_ is None:
                raise ValueError("warm_start requested, but model is undefined")
        else:
            from ..models.progressivegan import progressivegan
            from ..training import ProgressiveGANTrainer

            # Instantiate the generator and discriminator
            generator, discriminator = progressivegan(
                latent_size=self.latent_size,
                g_fmap_base=self.g_fmap_base,
                d_fmap_base=self.d_fmap_base,
                num_channels=self.num_channels,
                dimensionality=self.dimensionality,
            )
            self.model_ = ProgressiveGANTrainer(
                generator=generator,
                discriminator=discriminator,
                gradient_penalty=True,
            )
            self.current_resolution_ = 0

        # instantiate a progressive training helper and compile with loss and optimizer
        def _compile():
            self.model_.compile(
                g_optimizer=g_optimizer(**g_opt_args),
                d_optimizer=d_optimizer(**d_opt_args),
                g_loss_fn=g_loss,
                d_loss_fn=d_loss,
            )

        print(self.model_.generator.summary())
        print(self.model_.discriminator.summary())

        for resolution, info in dataset_train.items():
            if resolution < self.current_resolution_:
                continue
            # create a train dataset with features for resolution
            dataset = get_dataset(
                file_pattern=info.get("file_pattern"),
                batch_size=info.get("batch_size"),
                num_parallel_calls=num_parallel_calls,
                volume_shape=(resolution, resolution, resolution),
                n_classes=1,
                scalar_label=True,
                normalizer=info.get("normalizer"),
            )

            # grow the networks by one (2^x) resolution
            if resolution > self.current_resolution_:
                self.model_.generator.add_resolution()
                self.model_.discriminator.add_resolution()

            if multi_gpu:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    _compile()
            else:
                _compile()

            steps_per_epoch = epochs // info.get("batch_size")
            # save_best_only is set to False as it is an adversarial loss
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                str(model_dir),
                save_weights_only=True,
                save_best_only=False,
                save_freq=10,
            )

            # Train at resolution
            print("Resolution : {}".format(resolution))

            print("Transition phase")
            self.model_.fit(
                dataset,
                phase="transition",
                resolution=resolution,
                steps_per_epoch=steps_per_epoch,  # necessary for repeat dataset
                callbacks=[model_checkpoint_callback],
            )

            print("Resolution phase")
            self.model_.fit(
                dataset,
                phase="resolution",
                resolution=resolution,
                steps_per_epoch=steps_per_epoch,
                callbacks=[model_checkpoint_callback],
            )
            self.current_resolution_ = resolution
            # save the final weights
            self.model_.save_weights(model_dir)
        return self

    def generate(self, return_latents=False):
        """generate a synthetic image using the trained model"""
        if self.model_ is None:
            raise ValueError("Model is undefined. Please train or load a model")
        import nibabel as nib
        import numpy as np

        latents = tf.random.normal((1, self.latent_size))
        img = self.model_.generator.generate(latents)["generated"]
        img = np.squeeze(img)
        img = nib.Nifti1Image(img.astype(np.uint16), np.eye(4))
        if return_latents:
            return img, latents
        else:
            return img
