## Data augmentation

_Nobrainer_ provides methods of augmenting volumetric data. Augmentation is useful
when the amount of data is low, and it can create more generalizable and robust
models. Other packages have implemented methods of augmenting volumetric data,
but _Nobrainer_ is unique in that its augmentation methods are written in pure
TensorFlow. This allows these methods to be part of serializable `tf.data.Dataset`
pipelines.

In practice, [@kaczmarj](https://github.com/kaczmarj) has found that augmentations
improve the generalizability of semantic segmentation models for brain extraction.
Augmentation also seems to improve transfer learning models. For example, a meningioma
model trained from a brain extraction model that employed augmentation performed
better than a meningioma model trained from a brain extraction model that did not
use augmentation.

### Random rigid transformation

A rigid transformation is one that allows for rotations, translations, and reflections.
_Nobrainer_ implements rigid transformations in pure TensorFlow. Please refer to
`nobrainer.transform.warp` to apply a transformation matrix to a volume. You can
also apply random rigid transformations to the data input pipeline. When creating
your `tf.data.Dataset` with `nobrainer.volume.get_dataset`, simply set `augment=True`,
and about 50% of volumes will be augmented with random rigid transformations. To
use the function directly, please refer to `nobrainer.volume.apply_random_transform`.
Features and labels are transformed in the same way. Features are interpolated
linearly, whereas labels are interpolated using nearest neighbor. Below is an
example of a random rigid transformation applied to features and labels. The mask
in the right-hand column is a brain mask. Note that the MRI scan and brain mask
are transformed in the same way.

![Example of rigidly transforming features and labels volumes](https://user-images.githubusercontent.com/17690870/56315311-5ccaf580-6125-11e9-866a-af47aa76161c.png)
