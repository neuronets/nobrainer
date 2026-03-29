"""Bayesian VNet training script for autoresearch.

The autoresearch loop patches the ``# CONFIG:`` comment line below to
update hyperparameters between experiments.  On completion, this script
writes ``val_dice.json`` in the working directory.

Usage
-----
    python train_bayesian_vnet.py
"""

# CONFIG: {"learning_rate": 1e-4, "batch_size": 4, "n_epochs": 20, "kl_weight": 1e-4, "dropout_rate": 0.0}  # noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from monai.metrics import DiceMetric
from monai.utils import set_determinism
import torch
import torch.optim as optim

from nobrainer.dataset import get_dataset
from nobrainer.losses import dice as dice_loss_fn
from nobrainer.losses import elbo
from nobrainer.models.bayesian import BayesianVNet
from nobrainer.training import get_device


def main() -> None:
    # ------------------------------------------------------------------ #
    # Load config from script comment (patched by autoresearch loop)      #
    # ------------------------------------------------------------------ #
    script_text = Path(__file__).read_text()
    config: dict = {}
    for line in script_text.splitlines():
        if line.strip().startswith("# CONFIG:"):
            config = json.loads(line.split("# CONFIG:", 1)[1].strip())
            break

    lr: float = config.get("learning_rate", 1e-4)
    batch_size: int = int(config.get("batch_size", 4))
    n_epochs: int = int(config.get("n_epochs", 20))
    kl_weight: float = config.get("kl_weight", 1e-4)

    set_determinism(seed=42)
    device = get_device()

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #
    manifest_path = Path("data_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError("data_manifest.json not found. Run prepare.py first.")
    manifest = json.loads(manifest_path.read_text())
    train_images = manifest["train"]
    val_images = manifest["val"]
    label_suffix = "_label"  # adjust per dataset convention
    train_labels = [
        p.replace(".nii.gz", f"{label_suffix}.nii.gz") for p in train_images
    ]
    val_labels = [p.replace(".nii.gz", f"{label_suffix}.nii.gz") for p in val_images]

    train_loader = get_dataset(
        image_paths=train_images,
        label_paths=train_labels,
        batch_size=batch_size,
        augment=True,
        num_workers=0,
        cache_rate=0.0,
    )
    val_loader = get_dataset(
        image_paths=val_images,
        label_paths=val_labels,
        batch_size=1,
        num_workers=0,
        cache_rate=0.0,
    )

    # ------------------------------------------------------------------ #
    # Model, optimiser, metrics                                            #
    # ------------------------------------------------------------------ #
    model = BayesianVNet(n_classes=2, in_channels=1, kl_weight=kl_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = dice_loss_fn(softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    import pyro

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            optimizer.zero_grad()
            with pyro.poutine.trace():
                preds = model(imgs)
            labels_onehot = torch.zeros_like(preds)
            labels_onehot.scatter_(1, labels, 1.0)
            recon = recon_loss_fn(preds, labels_onehot)
            loss = elbo(model, kl_weight, recon)
            loss.backward()
            optimizer.step()

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            with pyro.poutine.trace():
                preds = model(imgs)
            preds_bin = torch.argmax(preds, dim=1, keepdim=True)
            dice_metric(preds_bin, labels)

    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    # ------------------------------------------------------------------ #
    # Write val_dice.json                                                  #
    # ------------------------------------------------------------------ #
    Path("val_dice.json").write_text(json.dumps({"val_dice": val_dice}))
    print(f"val_dice: {val_dice:.4f}")


if __name__ == "__main__":
    main()
