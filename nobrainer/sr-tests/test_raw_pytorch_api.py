"""Tests for the raw PyTorch API without the estimator layer."""

import nibabel as nib
import torch

import nobrainer.models
from nobrainer.prediction import predict
from nobrainer.training import fit as training_fit


class TestRawPyTorchAPI:
    """Test using raw nobrainer modules directly (no estimator)."""

    def test_raw_train_predict_cycle(self, train_eval_split, tmp_path):
        """Train with nobrainer.training.fit, predict with nobrainer.prediction.predict."""
        train_data, eval_pair = train_eval_split
        eval_img_path = eval_pair[0]

        # Build model directly
        model_factory = nobrainer.models.get("unet")
        model = model_factory(n_classes=2, channels=(4, 8), strides=(2,))

        # Build dataset directly
        from nobrainer.dataset import get_dataset

        image_paths = [pair[0] for pair in train_data]
        label_paths = [pair[1] for pair in train_data]

        loader = get_dataset(
            image_paths=image_paths,
            label_paths=label_paths,
            block_shape=(16, 16, 16),
            batch_size=2,
            binarize_labels=True,
        )

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        result = training_fit(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            max_epochs=2,
            gpus=0,
        )

        assert "final_loss" in result
        assert result["epochs_completed"] == 2

        # Predict
        model.eval()
        prediction = predict(
            inputs=eval_img_path,
            model=model,
            block_shape=(16, 16, 16),
        )

        assert isinstance(prediction, nib.Nifti1Image)
        assert len(prediction.shape) >= 3
