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

Models can be trained on neuroimaging volumes on the command line or with a Python script. All of the examples can be run within the nobrainer container.

Training data pre-requisites:
  1. Volumes must all be the same shape.
  2. Volumes must be in a format supported by [nibabel](http://nipy.org/nibabel/).
  3. Feature and label data must be available (e.g., T1 and aparc+aseg).

Examples:
  - [Command-line interface](#command-line-interface)
  - [Python interface](#python-interface)

Training progress can be visualized with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard):

```
$ singularity exec --clean-env --bind /path/to/models:/models nobrainer.sqsh \
    tensorboard --logdir /models
```

Note: `$` indicates a command-line call.


### Command-line interface

Run the following to show the script's help message

```shell
$ nobrainer train --help
```

The command below trains a HighRes3DNet model to perform brain extraction on T1s. Data are augmented in several ways (e.g., rotation, flipping, addition of random noise), and periodic evaluation is performed on the files listed in `eval-csv`.

```shell
$ nobrainer \
  --n-classes=2 \
  --model=highres3dnet \
  --model-opts='{"one_batchnorm_per_resblock": true, "dropout_rate": 0.25}' \
  --model-dir=/om/user/jakubk/nobrainer-models/models/brain-extraction/modeldir \
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
  --csv=/om/user/jakubk/nobrainer-models/data/features_labels.csv \
  --binarize \
  --eval-csv=/om/user/jakubk/nobrainer-models/data/evaluation_data.csv \
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

Please see the command-line help message for a description of these options. Explanations of the less intuitive options are provided below.

- `--model-opts`: model-specific options, which can be found in the model `__init__` signatures. This must be in JSON format.
- `--model-dir`: the directory in which to save model checkpoints. These checkpoints can be used later for inference and to create a `.pb` model file.
- `--multi-gpu`: if specified, train across all available GPUs. The batch will be split evenly across the GPUs. For example, if 8 GPUs are available and a batch of 8 is specified, each GPU receives a batch of size 1.
- `--prefetch`: prepare this many full volumes while training is going on. This can be especially useful when applying multiple augmentations, as each augmentation takes time.
- `--volume-shape`: shape of feature and label volumes.
- `--block-shape`: shape of blocks to take from the data. Most models cannot be trained on 256**3 inputs (exceeds available memory).
- `--strides`: number of voxels to stride across the volume in each dimension when yielding subsequent blocks. If `--strides` is equal to `--block-shape`, non-overlapping blocks will be yielded. If `--strides` is less than `--block-shape`, however, overlapping blocks will be yielded for training. Evaluation always uses non-overlapping blocks.
- `--csv`: path to CSV file with columns of features and labels filepaths. The file must have a header, although the column names can be arbitrary. For example:
```
features,labels
/absolute/path/to/0/T1.mgz,/absolute/path/to/0/aparc+aseg.mgz
/absolute/path/to/1/T1.mgz,/absolute/path/to/1/aparc+aseg.mgz
/absolute/path/to/2/T1.mgz,/absolute/path/to/2/aparc+aseg.mgz
```
- `--eval-csv`: path to CSV file with columns of features and labels filepaths, like the CSV file in `--csv`. The file must have a header, although the column names can be arbitrary. If this file is given, the model will be evaluated on these data periodically. Results are available in TensorBoard. Of course, the evaluation CSV should not include files from the training CSV.
- `--brainmask`: if specified, binarize the labels. If an aparc+aseg file is passed in as a label, the data will be binarized to create a brainmask.
- `--label-mapping`: a path to a CSV file mapping the original labels in the label volumes to a range `[0, n_classes-1]`. The file must have a header but the column names can be arbitrary. The original labels are in the first column, and the new labels are in the second column. Values in the labels not included in the second column are automatically made 0. For example, the file below is used in a 3-class classification problem, where values 10 and 50 correspond to one class, values 20 and 60 correspond to another class, and all other values correspond to another class.
```
original,new
10,1
50,1
20,2
60,2
```


### Python interface

Models can also be trained with nobrainer's Python interface. Trained models can be saved as `.pb` files in Python as well. These models can be used for prediction in other TensorFlow interfaces (JavaScript, C++, etc.).

The example below trains a HighRes3DNet model to perform brain extraction on 3D volumes. Training data are augmented in various ways. After training, the model is saved as a `.pb` model file and is used for prediction on new data.

```python
import nobrainer
import tensorflow as tf

# Instantiate object to perform real-time data augmentation on training data.
# This object is similar to `keras.preprocessing.image.ImageDataGenerator` but
# works with volumetric data.
volume_data_generator = nobrainer.VolumeDataGenerator(
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

# Instantiate TensorFlow model.
model = nobrainer.HighRes3DNet(
    n_classes=2,
    optimizer='Adam',
    learning_rate=0.01,
    one_batchnorm_per_resblock=True,
    dropout_rate=0.25)

# Read in filepaths to features and labels.
filepaths = nobrainer.read_csv("path/to/feature_labels.csv")

# Train model.
block_shape = (128, 128, 128)
nobrainer.train(
    model=model,
    volume_data_generator=volume_data_generator,
    filepaths=filepaths,
    volume_shape=(256, 256, 256),
    block_shape=block_shape,
    strides=block_shape,
    batch_size=1,
    n_epochs=1,
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


# Predict on a T1 using the saved model.
## Load saved model.
predictor = tf.contrib.predictor.from_saved_model("path/to/savedmodel")
predictions = nobrainer.predict_filepath(
    predictor=predictor,
    filepath="path/to/T1.mgz")
```
