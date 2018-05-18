"""Example script to predict using a SavedModel."""

import nibabel as nib
import nobrainer

# Predict on a T1 using the saved model. This returns a nibabel image object.
predicted = nobrainer.predict(
    inputs="T1.mgz",
    predictor="savedmodel",
    block_shape=(128, 128, 128))

# Save as .mgz and .nii.gz
nib.save(predicted, "brainmask.mgz")
nib.save(predicted, "brainmask.nii.gz")
