# Train

Below is a command-line call to train a HighRes3DNet model for brain extraction. Please see the command-line help message for a description of these options. Explanations of the less intuitive options are provided below.

- `--model-opts`: model-specific options, which can be found in the model `__init__` signatures. This must be in JSON format.
- `--model-dir`: the directory in which to save model checkpoints. These checkpoints can be used later for inference and to create a `.pb` model file.
- `--multi-gpu`: if specified, train across all available GPUs. The batch will be split evenly across the GPUs. For example, if 8 GPUs are available and a batch of 8 is specified, each GPU receives a batch of size 1.
- `--prefetch`: prepare this many full volumes while training is going on. This can be especially useful when applying multiple augmentations, as each augmentation takes time.
- `--volume-shape`: shape of feature and label volumes.
- `--block-shape`: shape of blocks to take from the data. Most models cannot be trained on 256**3 inputs (exceeds available memory).
- `--strides`: number of voxels to stride across the volume in each dimension when yielding subsequent blocks. If `--strides` is equal to `--block-shape`, non-overlapping blocks will be yielded. If `--strides` is less than `--block-shape`, however, overlapping blocks will be yielded for training. Evaluation always uses non-overlapping blocks.
- `--csv`: path to CSV file with columns of features and labels filepaths. The file must have a header, although the column names can be arbitrary. For example:
- `--eval-csv`: path to CSV file with columns of features and labels filepaths, like the CSV file in `--csv`. The file must have a header, although the column names can be arbitrary. If this file is given, the model will be evaluated on these data periodically. Results are available in TensorBoard. Of course, the evaluation CSV should not include files from the training CSV.
- `--brainmask`: binarize the labels. If an aparc+aseg file is passed in as a label, the data will be binarized to create a brainmask.

```
nobrainer train \
  --n-classes=2 \
  --model=highres3dnet \
  --model-opts='{"one_batchnorm_per_resblock": true, "dropout_rate": 0.25}' \
  --model-dir=modeldir \
  --optimizer=Adam \
  --learning-rate=0.01 \
  --batch-size=8 \
  --n-epochs=1 \
  --multi-gpu \
  --prefetch=8 \
  --save-summary-steps=25 \
  --save-checkpoints-steps=100 \
  --keep-checkpoint-max=500 \
  --volume-shape 256 256 256 \
  --block-shape 128 128 128 \
  --strides 64 64 64 \
  --csv=features_labels.csv \
  --binarize \
  --eval-csv=evaluation_data.csv \
  --samplewise-minmax \
  --rot90-x \
  --rot90-y \
  --rot90-z \
  --flip-x \
  --flip-y \
  --flip-z \
  --reduce-contrast \
  --salt-and-pepper \
  --gaussian
```


# Save as SavedModel

Convert the latest checkpoint and graph in `modeldir` (created during training) to a SavedModel (`.pb`). This SavedModel can be used in other TensorFlow interfaces (e.g., JavaScript, Go, C++, etc.).

```
nobrainer save \
  --model=highres3dnet \
  --model-opts='{"one_batchnorm_per_resblock": true, "dropout_rate": 0.25}' \
  --n-classes=2 \
  --block-shape 128 128 128 \
  --model-dir=modeldir \
  savedmodel
```


# Predict

This call will use the SavedModel to predict a brainmask from a T1.

```
nobrainer predict --block-shape 128 128 128 --model=savedmodel T1.mgz brainmask.mgz
```
