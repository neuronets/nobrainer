import tensorflow as tf

from . import losses, metrics
from .dataset import get_dataset, get_steps_per_epoch


class base_nobrainer_api:
    pass


class Segmentation(base_nobrainer_api):

    """
    Segmentation API.
    Sequentially applies transforms, train, predict and evaluate the segmentation model.

    """

    def __init__(
        self,
        base_model,
        train_pattern,
        train_n_volumes,
        evaluate_pattern,
        evaluate_n_volumes,
        n_classes,
        volume_shape,
        block_shape,
        augment,
        activation="relu",
        batchnorm=False,
        batch_size=None,
        multi_gpu=False,
        learning_rate=1e-04,
        loss=losses.dice,
        metrics=metrics.dice,
        **kwargs,
    ):
        self.base_model = base_model
        self.train_pattern = train_pattern
        self.train_n_volumes = train_n_volumes
        self.evaluate_pattern = evaluate_pattern
        self.evaluate_n_volumes = evaluate_n_volumes
        self.n_classes = n_classes
        self.volume_shape = volume_shape
        self.block_shape = block_shape
        self.augment = augment
        self.activation = activation
        self.batchnorm = batchnorm
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate

    def transform(self):
        raise NotImplementedError

    # train the model
    def fit(
        self,
        loss=losses.dice,
        metrics=[metrics.dice, metrics.jaccard],
        learning_rate=1e-04,
        epochs=1,
        checkpoint_dir="./",
        **kwargs,
    ):

        # load data
        self.train_dataset = get_dataset(
            file_pattern=self.train_pattern,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            volume_shape=self.volume_shape,
            block_shape=self.block_shape,
            augment=self.augment,
            scalar_label=kwargs["scalar_label"],
            n_epochs=kwargs["n_epochs"],
            mapping=kwargs["mapping"],
            normalizer=kwargs["normalizer"],
            shuffle_buffer_size=kwargs["shuffle_buffer_size"],
            num_parallel_calls=kwargs["num_parallel_calls"],
        )

        self.evaluate_dataset = get_dataset(
            file_pattern=self.evaluate_pattern,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            volume_shape=self.volume_shape,
            block_shape=self.block_shape,
            augment=self.augment,
            scalar_label=kwargs["scalar_label"],
            n_epochs=kwargs["n_epochs"],
            mapping=kwargs["mapping"],
            normalizer=kwargs["normalizer"],
            shuffle_buffer_size=kwargs["shuffle_buffer_size"],
            num_parallel_calls=kwargs["num_parallel_calls"],
        )

        # Instantiate and compile the model
        def _create_model(self, **kwargs):
            self.model = self.base_model(
                n_classes=self.n_classes,
                input_shape=(*self.input_shape, 1),
                activation=self.activation,
                batch_size=self.batch_size,
                **kwargs,
            )
            self.model.compile(
                tf.keras.optimizers.Adam(learning_rate),
                loss=loss,
                metrics=metrics,
            )
            print(self.model.summary())
            return self.model

        # estimate or train a model
        if self.multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = _create_model(self)
        else:
            self.model = _create_model(self)

        self.train_steps = get_steps_per_epoch(
            n_volumes=self.train_n_volumes,
            volume_shape=self.volume_shape,
            block_shape=self.block_shape,
            batch_size=self.batch_size,
        )

        self.evaluate_steps = get_steps_per_epoch(
            n_volumes=self.train_n_volumes,
            volume_shape=self.volume_shape,
            block_shape=self.block_shape,
            batch_size=self.batch_size,
        )

        # TODO add checkpoint
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            steps_per_epoch=self.train_steps,
            validation_data=self.evaluate_dataset,
            validation_steps=self.evaluate_steps,
        )

        return self.model, self.history

    def fit_transform(self):
        raise NotImplementedError

    def save(self, file_path, **kwargs):
        """Saves a trained model"""

        self.model.save(file_path, **kwargs)

    def predict(self, test_data):
        """Makes aprediction using trained model"""
        # from .prediction import predict
        self.out = self.model.predict(test_data)

        return self.out
