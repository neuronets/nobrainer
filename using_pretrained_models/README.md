# Use pre-trained models to segment brains in MR images

- [Pre-trained Keras models](https://keras.io/applications/#usage-examples-for-image-classification-models)
- [Pre-trained TensorFlow models](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)
- [TensorFlow model zoo](https://github.com/tensorflow/models)


Command to run jupyter notebook from the tensorflow-gpu singularity image: 

```shell
$> singularity shell -B /om/user/jakubk/neuro_nn/data:/data -B /home/jakubk/neuro_nn:/home/neuro_nn /om/user/jakubk/singularity_images/satra-om-images-keras-gpu.img
$> unset XDG_RUNTIME_DIR
$> jupyter notebook --ip=* --port=9000
```

(Is there a way to override the runscript? Similar to `docker run --entrypoint="jupyter notebook" <image>)`


## ImageNet networks

### [VGG16](https://keras.io/applications/#vgg16)

- [example using pre-trained VGG16](https://github.com/fchollet/keras/issues/4465#issuecomment-262229784)
- [another example using pre-trained VGG16](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069)



## [DeepMedic](https://github.com/Kamnitsask/deepmedic)

- Processes NIfTI images only.
