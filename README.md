# Nobrainer

[![Build Status](https://travis-ci.com/kaczmarj/nobrainer.svg?branch=master)](https://travis-ci.com/kaczmarj/nobrainer)

_Nobrainer_ is a deep learning framework for 3D image processing. It implements several 3D convolutional models from recent literature, methods for loading and augmenting volumetric data that can be used with any TensorFlow or Keras model, losses and metrics for 3D data, and simple utilities for model training, evaluation, prediction, and transfer learning.

Soon, _Nobrainer_ will also provide pre-trained models for brain extraction, brain segmentation, and other tasks.

The _Nobrainer_ project is supported by NIH R01 EB020470 and is distributed under the Apache 2.0 license.

## Guide Jupyter Notebooks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaczmarj/nobrainer)

Please refer to the Jupyter notebooks in the [guide](/guide) directory to get started with _Nobrainer_. [Try them out](https://colab.research.google.com/github/kaczmarj/nobrainer) in Google Collaboratory!

## Installing Nobrainer

### Container

We recommend using the official _Nobrainer_ Docker container, which includes all of the dependencies necessary to use the framework. Please see the available images on [DockerHub](https://hub.docker.com/r/kaczmarj/nobrainer)

#### GPU support

The _Nobrainer_ containers with GPU support use CUDA 10, which requires Linux NVIDIA drivers `>=410.48`.

```
$ docker pull kaczmarj/nobrainer:latest-gpu
$ singularity pull docker://kaczmarj/nobrainer:latest-gpu
```

#### CPU only

```
$ docker pull kaczmarj/nobrainer:latest
$ singularity pull docker://kaczmarj/nobrainer:latest
```

### pip

_Nobrainer_ can also be installed with pip. Use the extra `[gpu]` to install TensorFlow with GPU support and the `[cpu]` extra to install TensorFlow without GPU support.

```
$ pip install --no-cache-dir nobrainer[gpu]
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
