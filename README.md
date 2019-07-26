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
  - [Transfer learning](#transfer-learning)
    - [Example](#example)
- [Data augmentation](#data-augmentation)
  - [Random rigid transformation](#random-rigid-transformation)
- [Package layout](#package-layout)
- [Questions or issues](#questions-or-issues)

## Guide Jupyter Notebooks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuronets/nobrainer)

Please refer to the Jupyter notebooks in the [guide](/guide) directory to get started with _Nobrainer_. [Try them out](https://colab.research.google.com/github/neuronets/nobrainer) in Google Collaboratory!

## Installation

### Container

We recommend using the official _Nobrainer_ Docker container, which includes all of the dependencies necessary to use the framework. 
If you are new to Docker, [you can start here](https://www.docker.com/get-started))
Please see the available _Nobrainer_ images on [DockerHub](https://hub.docker.com/r/kaczmarj/nobrainer). 

#### GPU support

The _Nobrainer_ containers with GPU support use CUDA 10, which requires Linux NVIDIA drivers `>=410.48`. These drivers are not included in the container, but can be obtained from [here](https://www.nvidia.com/object/unix.html)

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

Pre-trained networks are available in the [_Nobrainer_ models](https://github.com/neuronets/nobrainer-models/releases) repository as releases and can be downloaded into the working directory. 

```bash
# Specify release version and model name
git clone https://github.com/neuronets/nobrainer-models
RELEASE=0.1
MODEL_NAME=brain-extraction-unet-128iso-model.h5
wget https://github.com/neuronets/nobrainer-models/releases/download/$RELEASE/$MODEL_NAME -P nobrainer-models
```

### Predicting a brainmask for a T1-weighted brain scan

In the following examples, we will use a 3D U-Net trained for brain extraction and documented in [_Nobrainer_ models](https://github.com/neuronets/nobrainer-models#brain-extraction).

In the base case, we run the T1w scan through the model for prediction using Docker in the working directory.

```bash
# Get sample T1w scan and run docker
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/data/nobrainer-models/brain-extraction-unet-128iso-model.h5 \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask.nii.gz
```

For binary segmentation where we expect one predicted region, as is the case with brain extraction, we can reduce false positives by removing all predictions not connected to the largest contiguous label.

```bash
# Get sample T1w scan and run docker
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/data/nobrainer-models/brain-extraction-unet-128iso-model.h5 \
    --largest-label \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-largestlabel.nii.gz
```

Because the network was trained on randomly rotated data, it should be agnostic to orientation. Therefore, we can rotate the volume, predict on it, undo the rotation in the prediction, and average the prediction with that from the original volume. This can lead to a better overall prediction but will at least double the processing time. To enable this, use the flag `--rotate-and-predict` in `nobrainer predict`.

```bash
# Get sample T1w scan and run docker
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/data/nobrainer-models/brain-extraction-unet-128iso-model.h5 \
    --rotate-and-predict \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-withrotation.nii.gz
```

Combining the above, we can usually achieve the best brain extraction by using `--rotate-and-predict` in conjunction with `--largest-label`.

```bash
# Get sample T1w scan and run docker
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data kaczmarj/nobrainer \
  predict \
    --model=/data/nobrainer-models/brain-extraction-unet-128iso-model.h5 \
    --largest-label \
    --rotate-and-predict \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-maybebest.nii.gz
```

### Transfer learning

The pre-trained models can be used for transfer learning. To avoid forgetting important information in the pre-trained model, you can apply regularization to the kernel weights and also use a low learning rate. For more information, please see the [_Nobrainer_ guide notebook](/guide/transfer_learning.ipynb) on transfer learning.

#### Example

As an example of transfer learning, [@kaczmarj](https://github.com/kaczmarj) re-trained a brain extraction model to label meningiomas in 3D T1-weighted, contrast-enhanced MR scans. The original model is publicly available and was trained on 10,000 T1-weighted MR brain scans from healthy participants. These were all research scans (i.e., non-clinical) and did not include any contrast agents. The meningioma dataset, on the other hand, was composed of relatively few scans, all of which were clinical and used gadolinium as a contrast agent. You can observe the differences in contrast below.

![Brain extraction model prediction](https://github.com/kaczmarj/nobrainer-models/blob/master/images/brain-extraction/unet-best-prediction.png?raw=true)
![Meningioma extraction model prediction](https://user-images.githubusercontent.com/17690870/55470578-e6cb7800-55d5-11e9-991f-fe13c03ab0bd.png)

Despite the differences between the two datasets, transfer learning led to a much better model than training from randomly-initialized weights. As evidence, please see below violin plots of Dice coefficients on a validation set. In the left plot are Dice coefficients of predictions obtained with the model trained from randomly-initialized weights, and on the right are Dice coefficients of predictions obtained with the transfer-learned model. In general, Dice coefficients are higher on the right, and the variance of Dice scores is lower. Overall, the model on the right is more accurate and more robust than the one on the left.

<div style="text-align: center">
<img src="https://user-images.githubusercontent.com/17690870/56313232-1e7f0780-6120-11e9-8f1a-62b8c3d48e15.png" alt="" width="49%" />
<img src="https://user-images.githubusercontent.com/17690870/56313239-23dc5200-6120-11e9-88eb-0e9ebca6ba83.png" alt="" width="49%" />
</div>

## Data augmentation

_Nobrainer_ provides methods of augmenting volumetric data. Augmentation is useful when the amount of data is low, and it can create more generalizable and robust models. Other packages have implemented methods of augmenting volumetric data, but _Nobrainer_ is unique in that its augmentation methods are written in pure TensorFlow. This allows these methods to be part of serializable `tf.data.Dataset` pipelines and used for training on TPUs.

In practice, [@kaczmarj](https://github.com/kaczmarj) has found that augmentations improve the generalizability of semantic segmentation models for brain extraction. Augmentation also seems to improve transfer learning models. For example, a meningioma model trained from a brain extraction model that employed augmentation performed better than a meningioma model trained from a brain extraction model that did not use augmentation.

### Random rigid transformation

A rigid transformation is one that allows for rotations, translations, and reflections. _Nobrainer_ implements rigid transformations in pure TensorFlow. Please refer to `nobrainer.transform.warp` to apply a transformation matrix to a volume. You can also apply random rigid transformations to the data input pipeline. When creating you `tf.data.Dataset` with `nobrainer.volume.get_dataset`, simply set `augment=True`, and about 50% of volumes will be augmented with random rigid transformations. To use the function directly, please refer to `nobrainer.volume.apply_random_transform`. Features and labels are transformed in the same way. Features are interpolated linearly, whereas labels are interpolated using nearest neighbor. Below is an example of a random rigid transformation applied to features and labels. The mask in the right-hand column is a brain mask. Note that the MRI scan and brain mask are transformed in the same way.

![Example of rigidly transforming features and labels volumes](https://user-images.githubusercontent.com/17690870/56315311-5ccaf580-6125-11e9-866a-af47aa76161c.png)

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
