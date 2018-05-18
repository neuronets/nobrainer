# nobrainer

Neural networks for brain extraction and brain labelling from structural magnetic resonance images.

_Note: this is a work in progress._

## Examples

Please see the [examples](examples) directory.


## Getting started

### Get the container

```shell
$ docker pull kaczmarj/nobrainer
# or
$ singularity build nobrainer.sqsh docker://kaczmarj/nobrainer
```

### Train your own models

Models can be trained on neuroimaging volumes on the command line or with a Python script. All of the examples can be run within the _Nobrainer_ container. Please see the [examples](examples) for more information.

Training data pre-requisites:
  1. Volumes must be in a format supported by [nibabel](http://nipy.org/nibabel/).
  2. Feature and label data must be available (e.g., T1 and aparc+aseg).

Training progress can be visualized with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard):

```
$ singularity exec --clean-env --bind /path/to/models:/models nobrainer.sqsh \
    tensorboard --logdir /models
```

### Predict using trained models

We are in the process of training robust models for brain extraction and brain labelling. Stay tuned for information on how to use these models.
