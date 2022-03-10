# import importlib
import os
from pathlib import Path

import tensorflow as tf

from .base import BaseEstimator
from .. import losses
from ..dataset import get_dataset
from ..models.progressivegan import progressivegan
from ..training import ProgressiveGANTrainer


class ProgressiveGeneration(BaseEstimator):
    """Perform generation type operations"""

    state_variables = ["resolution_batch_size_map_"]

    def __init__(
        self,
        latent_size=256,
        label_size=0,
        num_channels=1,
        dimensionality=3,
        g_fmap_base=1024,
        d_fmap_base=1024,
        # g_optimizer=None,
        # g_opt_args=None,
        # d_optimizer=None,
        # d_opt_args=None,
        # g_loss=losses.wasserstein,
        # d_loss=losses.wasserstein,
        # resolution_batch_size_map= None,
        # input_file_pattern="*res-%03d*.tfrec"
    ):
        self.model_ = progressivegan
        self.latent_size = latent_size
        self.label_size = label_size
        self.g_fmap_base = g_fmap_base
        self.d_fmap_base = d_fmap_base
        self.num_channels = num_channels
        self.dimensionality = dimensionality

        # self.g_optimizer = g_optimizer
        # self.g_opt_args = g_opt_args or {}
        # self.d_optimizer = d_optimizer
        # self.d_opt_args = d_opt_args or {}
        # self.g_loss = g_loss
        # self.d_loss = d_loss
        # self.file_pattern = input_file_pattern
        # self.resolution_batch_size_map_ = resolution_batch_size_map or {}
        # self.resolutions_ = sorted(list(self.resolution_batch_size_map_.keys()))

    def fit(
        self,
        epochs=2,
        checkpoint_dir=os.getcwd(),
        # TODO: figure out whether optimizer args should be flattened
        g_optimizer=None,
        g_opt_args=None,
        d_optimizer=None,
        d_opt_args=None,
        g_loss=losses.wasserstein,
        d_loss=losses.wasserstein,
        resolution_batch_size_map=None,
        input_file_pattern="*res-%03d*.tfrec",
        warm_start=False,
        multi_gpu=False,
        num_parallel_calls=None,
    ):
        """Train a progressive gan model"""
        # TODO: check validity of datasets

        n_classes = 1  # dummy labels as this is unsupervised training
        self.resolution_batch_size_map_ = resolution_batch_size_map or {}
        self.resolutions_ = sorted(list(self.resolution_batch_size_map_.keys()))

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

        def _create():
            # Instantiate the generator and discriminator
            self.generator_, self.discriminator_ = self.model_(
                latent_size=self.latent_size,
                g_fmap_base=self.g_fmap_base,
                d_fmap_base=self.d_fmap_base,
            )

        def _compile():
            # instantiate a progressive training helper and compile with loss and optimizer
            self.pgan_trainer = ProgressiveGANTrainer(
                generator=self.generator_,
                discriminator=self.discriminator_,
                gradient_penalty=True,
            )

            self.pgan_trainer.compile(
                g_optimizer=self.g_optimizer(**self.g_opt_args),
                d_optimizer=self.d_optimizer(**self.d_opt_args),
                g_loss_fn=self.g_loss,
                d_loss_fn=self.d_loss,
            )

        # without warmstart and multi gpu for now
        # mod = importlib.import_module("..models", "nobrainer.processing")
        # base_model = getattr(mod, self.base_model)
        _create()
        print(self.generator_.summary())
        print(self.discriminator_.summary())

        for resolution in self.resolutions_:

            # create a train dataset with features for resolution
            dataset_train = get_dataset(
                file_pattern=self.file_pattern % (resolution),
                batch_size=self.resolution_batch_size_map_[resolution],
                num_parallel_calls=num_parallel_calls,
                volume_shape=(resolution, resolution, resolution),
                n_classes=n_classes,
                scalar_label=True,
                normalizer=None,
            )

            # grow the networks by one (2^x) resolution
            self.generator_.add_resolution()
            self.discriminator_.add_resolution()

            # Note: multigpu should be added here
            # compile the trainers
            _compile()

            steps_per_epoch = epochs // self.resolution_batch_size_map_[resolution]
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
            self.pgan_trainer.fit(
                dataset_train,
                phase="transition",
                resolution=resolution,
                steps_per_epoch=steps_per_epoch,  # necessary for repeat dataset
                callbacks=[model_checkpoint_callback],
            )

            print("Resolution phase")
            self.pgan_trainer.fit(
                dataset_train,
                phase="resolution",
                resolution=resolution,
                steps_per_epoch=steps_per_epoch,
                callbacks=[model_checkpoint_callback],
            )

            # save the final weights
            self.generator_.save(
                str(model_dir.joinpath("generator_res_{}".format(resolution)))
            )

        return self

    def generate(
        self,
        x,
    ):
        """generate a synthetic image using the trained model"""
        if self.model_ is None:
            raise ValueError("Model is undefined. Please train or load a model")
        import nibabel as nib
        import numpy as np

        latents = tf.random.normal((1, self.latent_size))
        generate = self.generator_.signatures["serving_default"]
        img = generate(latents)["generated"]
        img = np.squeeze(img)
        img = nib.Nifti1Image(img.astype(np.uint16), np.eye(4))
        return img, latents
