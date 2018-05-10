# nobrainer

Neural networks for brain extraction and brain parcellation from structural MR images.

_Note: this is a work in progress._


## Get the container

```shell
$ docker pull kaczmarj/nobrainer
# or
$ singularity build nobrainer.sqsh docker://kaczmarj/nobrainer
```


## Train your own models

Models can be trained on neuroimaging volumes. Please see examples below. All of the examples can be run within the nobrainer container.

Note: `$` indicates a command-line call.

### Command-line interface

Data pre-requisites:
1. Volumes must all be the same shape.
2. Volumes must be in a format supported by [nibabel](http://nipy.org/nibabel/).
3. Feature and label data must be available (e.g., T1 and aparc+aseg).

Run the following to show the script's help message

```shell
$ train_on_volumes.py --help
```

The command below trains a HighRes3DNet model to perform brain extraction on T1s.

```shell
$ train_on_volumes.py \
  --n-classes=2 \
  --model=highres3dnet \
  --brainmask \
  --optimizer=Adam \
  --learning-rate=0.001 \
  --batch-size=6 \
  --n-epochs=5 \
  --vol-shape 256 256 256 \
  --block-shape 64 64 64  \
  --strides 32 32 32 \
  --model-dir=path/to/model \
  --csv=features_labels.csv \
  --eval-csv=evaluation_data.csv \
  --multi-gpu
```

- `--brainmask` if specified, binarize the labels. If an aparc+aseg file is passed in as a label, the data will be binarized to create a brainmask.
- `--vol-shape`: shape of feature and label volumes.
- `--block-shape`: shape of blocks to take from the data. Most models cannot be trained on 256**3 inputs (exceeds available memory).
- `--strides`: number of voxels to stride across the volume in each dimension when yielding subsequent blocks. If `--strides` is equal to `--block-shape`, non-overlapping blocks will be yielded. If `--strides` is less than `--block-shape`, however, overlapping blocks will be yielded for training.
- `--csv`: path to CSV file with columns of features and labels filepaths. The file must have a header, although the column names can be arbitrary.
```
features,labels
path/to/0/T1.mgz,path/to/0/aparc+aseg.mgz
path/to/1/T1.mgz,path/to/1/aparc+aseg.mgz
path/to/2/T1.mgz,path/to/2/aparc+aseg.mgz
```
- `--eval-csv`: path to CSV file with columns of features and labels filepaths. The file must have a header, although the column names can be arbitrary. If this file is given, the model will be evaluated on these data periodically. Results are available in TensorBoard. The CSV should look like the CSV in `--csv`, but of course, use different files.
- `--multi-gpu`: if specified, train across multiple GPUs. The batch is split across GPUs, so the batch must be divisible by the number of GPUs.


Training progress can be visualized with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard):

```
singularity exec --bind /path/to/models:/models nobrainer.sqsh \
  tensorboard --logdir /models
```

If `--eval-csv` was specified, evaluation results will also appear in TensorBoard.


### Python interface

```python
import nobrainer
import tensorflow as tf

filepaths = nobrainer.read_csv("path/to/feature_labels.csv")

datagen = nobrainer.VolumeDataGenerator(
    samplewise_minmax=True,
    rot90_x=True,
    rot90_y=True,
    rot90_z=True,
    flip_x=True,
    flip_y=True,
    flip_z=True,
    salt_and_pepper=True,
    gaussian=True,
    reduce_contrast=True,
    binarize_y=True)

model = nobrainer.HighRes3DNet(
    n_classes=2,
    optimizer='Adam',
    learning_rate=0.01,
    one_batchnorm_per_resblock=True)

block_shape = (128, 128, 128)
nobrainer.train(
    model=model,
    volume_data_generator=datagen,
    filepaths=filepaths,
    volume_shape=(256, 256, 256),
    block_shape=block_shape,
    strides=block_shape,
    batch_size=1,
    num_epochs=1,
    prefetch=4)


# Save model to .pb file.
## The shape of the input volume (batch_size, *block_shape, 1). The 1 indicates
## one channel (grayscale).
volume = tf.placeholder(tf.float32, shape=[None, *block_shape, 1])
## Attach the volume placeholder above to the input function for the model.
serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
    'volume': volume})
## Save the model to a .pb file in the specified directory.
model.export_savedmodel("path/to/savedmodel", serving_input_fn)


# Predict on a T1 using saved model.
## Load saved model.
predictor = tf.contrib.predictor.from_saved_model("path/to/savedmodel")
## Load volume and block in preparation for prediction.
anat = nobrainer.read_volume("path/to/t1.nii.gz", dtype="float32")
anat = nobrainer.normalize_zero_one(features)
features = nobrainer.as_blocks(anat, block_shape)
## Add a dimension for single channel.
features = features[..., None]
## Run prediction. Returns a dictionary with tensors as values.
predictions = predictor({'volume': features})
## Combine the predicted blocks as a full volume.
predictions = nobrainer.from_blocks(predictions['class_ids'], (256, 256, 256))
```
