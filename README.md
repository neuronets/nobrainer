# Nobrainer

[![Build Status](https://travis-ci.com/neuronets/nobrainer.svg?branch=master)](https://travis-ci.com/neuronets/nobrainer)

![Model's prediction of brain mask](https://github.com/neuronets/nobrainer-models/blob/master/images/brain-extraction/unet-best-prediction.png?raw=true) ![Model's prediction of brain mask](https://github.com/neuronets/nobrainer-models/blob/master/images/brain-extraction/unet-worst-prediction.png?raw=true) <sub>__Figure__: In the first column are T1-weighted brain scans, in the middle are a trained model's predictions, and on the right are binarized FreeSurfer segmentations. Despite being trained on binarized FreeSurfer segmentations, the model outperforms FreeSurfer in the bottom scan, which exhibits motion distortion. It took about three seconds for the model to predict each brainmask using an NVIDIA GTX 1080Ti. It takes about 70 seconds on a recent CPU.</sub>

_Nobrainer_ is a deep learning framework for 3D image processing. It implements several 3D convolutional models from recent literature, methods for loading and augmenting volumetric data that can be used with any TensorFlow or Keras model, losses and metrics for 3D data, and simple utilities for model training, evaluation, prediction, and transfer learning.

_Nobrainer_ also provides pre-trained models for brain extraction, brain segmentation, and other tasks. Please see the [_Nobrainer_ models](https://github.com/neuronets/nobrainer-models) repository for more information.

The _Nobrainer_ project is supported by NIH R01 EB020470 and is distributed under the Apache 2.0 license.

## Table of contents

- [Guide Jupyter Notebooks](#guide-jupyter-notebooks-)
- [Installation](#installation)
  - [Container](#container)
    - [GPU support](#gpu-support)
    - [CPU only](#cpu-only)
  - [pip](#pip)
- [Using pre-trained networks](#using-pre-trained-networks)
  - [Predicting a brainmask for a T1-weighted brain scan](#predicting-a-brainmask-for-a-t1-weighted-brain-scan)
- [Package layout](#package-layout)
- [Questions or issues](#questions-or-issues)

## Guide Jupyter Notebooks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuronets/nobrainer)

Please refer to the Jupyter notebooks in the [guide](/guide) directory to get started with _Nobrainer_. [Try them out](https://colab.research.google.com/github/neuronets/nobrainer) in Google Collaboratory!

## Installation

### Container

We recommend using the official _Nobrainer_ Docker container, which includes all of the dependencies necessary to use the framework. Please see the available images on [DockerHub](https://hub.docker.com/r/kaczmarj/nobrainer)

#### GPU support

The _Nobrainer_ containers with GPU support use CUDA 10, which requires Linux NVIDIA drivers `>=410.48`. These drivers are not included in the container.

```
$ docker pull kaczmarj/nobrainer:latest-gpu
$ singularity pull docker://kaczmarj/nobrainer:latest-gpu
```

#### CPU only

This container can be used on all systems that have Docker or Singularity and does not require special hardware. This container, however, should not be used for model training (it will be very slow).

```
$ docker pull kaczmarj/nobrainer:latest
$ singularity pull docker://kaczmarj/nobrainer:latest
```

### pip

_Nobrainer_ can also be installed with pip. Use the extra `[gpu]` to install TensorFlow with GPU support and the `[cpu]` extra to install TensorFlow without GPU support. GPU support requires CUDA 10, which requires Linux NVIDIA drivers `>=410.48`.

```
$ pip install --no-cache-dir nobrainer[gpu]
```

## Using pre-trained networks

Pre-trained networks are available in the [_Nobrainer_ models](https://github.com/neuronets/nobrainer-models) repository. Prediction can be done on the command-line with `nobrainer predict` or in Python.

### Predicting a brainmask for a T1-weighted brain scan

In the following examples, we will use a 3D U-Net trained for brain extraction and documented in [_Nobrainer_ models](https://github.com/neuronets/nobrainer-models#brain-extraction).

In the base case, we run the T1w scan through the model for prediction.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/models/brain-extraction-unet-128iso-model.h5 \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask.nii.gz
```

For binary segmentation where we expect one predicted region, as is the case with brain extraction, we can reduce false positives by removing all predictions not connected to the largest contiguous label.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/models/brain-extraction-unet-128iso-model.h5 \
    --largest-label \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-largestlabel.nii.gz
```

Because the network was trained on randomly rotated data, it should be agnostic to orientation. Therefore, we can rotate the volume, predict on it, undo the rotation in the prediction, and average the prediction with that from the original volume. This can lead to a better overall prediction but will at least double the processing time. To enable this, use the flag `--rotate-and-predict` in `nobrainer predict`.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/models/brain-extraction-unet-128iso-model.h5 \
    --rotate-and-predict \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-withrotation.nii.gz
```

Combining the above, we can usually achieve the best brain extraction by using `--rotate-and-predict` in conjunction with `--largest-label`.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/models/brain-extraction-unet-128iso-model.h5 \
    --largest-label \
    --rotate-and-predict \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-maybebest.nii.gz
```

## Package layout

- `nobrainer.io`: input/output methods
- `nobrainer.layers`: custom layers, which conform to the Keras API
- `nobrainer.losses`: loss functions for volumetric segmentation
- `nobrainer.metrics`: metrics for volumetric segmentation
- `nobrainer.models`: pre-defined Keras models
- `nobrainer.training`: training utilities (supports training on single and multiple GPUs)
- `nobrainer.transform`: random rigid transformations for data augmentation
- `nobrainer.volume`: `tf.data.Dataset` creation and data augmentation utilities

## Questions or issues

If you have questions about _Nobrainer_ or encounter any issues using the framework, please submit a GitHub issue. If you have a feature request, we encourage you to submit a pull request.
