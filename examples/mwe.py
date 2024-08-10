import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras.layers as KL
from keras.models import Model
import numpy as np
import tensorflow as tf

from nobrainer.ext.SynthSeg.model_inputs import build_model_inputs
from nobrainer.ext.lab2im import edit_tensors as l2i_et
from nobrainer.ext.lab2im import layers, utils
from nobrainer.ext.lab2im.edit_volumes import get_ras_axes
from nobrainer.models.labels_to_image_model import get_shapes


def labels_to_image_model(
    labels_shape,
    n_channels,
    generation_labels,
    output_labels,
    n_neutral_labels,
    atlas_res,
    target_res,
    output_shape=None,
    output_div_by_n=None,
    flipping=True,
    aff=None,
    scaling_bounds=0.2,
    rotation_bounds=15,
    shearing_bounds=0.012,
    translation_bounds=False,
    nonlin_std=3.0,
    nonlin_scale=0.0625,
    randomise_res=False,
    max_res_iso=4.0,
    max_res_aniso=8.0,
    data_res=None,
    thickness=None,
    bias_field_std=0.5,
    bias_scale=0.025,
    return_gradients=False,
):
    # reformat resolutions
    labels_shape = utils.reformat_to_list(labels_shape)
    n_dims, _ = utils.get_dims(labels_shape)
    atlas_res = utils.reformat_to_n_channels_array(atlas_res, n_dims, n_channels)
    data_res = (
        atlas_res
        if data_res is None
        else utils.reformat_to_n_channels_array(data_res, n_dims, n_channels)
    )
    thickness = (
        data_res
        if thickness is None
        else utils.reformat_to_n_channels_array(thickness, n_dims, n_channels)
    )
    atlas_res = atlas_res[0]
    target_res = (
        atlas_res
        if target_res is None
        else utils.reformat_to_n_channels_array(target_res, n_dims)[0]
    )

    # get shapes
    crop_shape, output_shape = get_shapes(
        labels_shape, output_shape, atlas_res, target_res, output_div_by_n
    )

    # define model inputs
    labels_input = KL.Input(
        shape=labels_shape + [1], name="labels_input", dtype="int32"
    )
    means_input = KL.Input(
        shape=list(generation_labels.shape) + [n_channels], name="means_input"
    )
    stds_input = KL.Input(
        shape=list(generation_labels.shape) + [n_channels], name="std_devs_input"
    )
    list_inputs = [labels_input, means_input, stds_input]

    # # deform labels
    # labels = layers.RandomSpatialDeformation(
    #     scaling_bounds=scaling_bounds,
    #     rotation_bounds=rotation_bounds,
    #     shearing_bounds=shearing_bounds,
    #     translation_bounds=translation_bounds,
    #     nonlin_std=nonlin_std,
    #     nonlin_scale=nonlin_scale,
    #     inter_method="nearest",
    # )(labels_input)
    labels = labels_input

    # # cropping
    # if crop_shape != labels_shape:
    #     labels = layers.RandomCrop(crop_shape)(labels)

    # # flipping
    # if flipping:
    #     assert aff is not None, "aff should not be None if flipping is True"
    #     labels = layers.RandomFlip(
    #         get_ras_axes(aff, n_dims)[0], True, generation_labels, n_neutral_labels
    #     )(labels)

    # build synthetic image
    image = layers.SampleConditionalGMM(generation_labels)(
        [labels, means_input, stds_input]
    )

    # # apply bias field
    # if bias_field_std > 0:
    #     image = layers.BiasFieldCorruption(bias_field_std, bias_scale, False)(image)

    # # intensity augmentation
    # image = layers.IntensityAugmentation(
    #     clip=300, normalise=True, gamma_std=0.5, separate_channels=True
    # )(image)

    # # loop over channels
    # channels = list()
    # split = (
    #     KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image)
    #     if (n_channels > 1)
    #     else [image]
    # )

    channel = image
    # for i, channel in enumerate(split):
    if randomise_res:
        max_res_iso = np.array(
            utils.reformat_to_list(max_res_iso, length=n_dims, dtype="float")
        )
        max_res_aniso = np.array(
            utils.reformat_to_list(max_res_aniso, length=n_dims, dtype="float")
        )
        max_res = np.maximum(max_res_iso, max_res_aniso)
        resolution, blur_res = layers.SampleResolution(
            atlas_res, max_res_iso, max_res_aniso
        )(means_input)
        sigma = l2i_et.blurring_sigma_for_downsampling(
            atlas_res, resolution, thickness=blur_res
        )
        channel = layers.DynamicGaussianBlur(
            0.75 * max_res / np.array(atlas_res), 1.03
        )([channel, sigma])
        channel = layers.MimicAcquisition(atlas_res, atlas_res, output_shape, False)(
            [channel, resolution]
        )
        # channels.append(channel)

        # else:
        #     sigma = l2i_et.blurring_sigma_for_downsampling(
        #         atlas_res, data_res[i], thickness=thickness[i]
        #     )
        #     channel = layers.GaussianBlur(sigma, 1.03)(channel)
        #     resolution = KL.Lambda(
        #         lambda x: tf.convert_to_tensor(data_res[i], dtype="float32")
        #     )([])
        #     channel = layers.MimicAcquisition(atlas_res, data_res[i], output_shape)(
        #         [channel, resolution]
        #     )
        #     channels.append(channel)

    # # concatenate all channels back
    # image = (
    #     KL.Lambda(lambda x: tf.concat(x, -1))(channels)
    #     if len(channels) > 1
    #     else channels[0]
    # )

    image = channel

    # # compute image gradient
    # if return_gradients:
    #     image = layers.ImageGradients("sobel", True, name="image_gradients")(image)
    #     image = layers.IntensityAugmentation(clip=10, normalise=True)(image)

    # # resample labels at target resolution
    # if crop_shape != output_shape:
    #     labels = l2i_et.resample_tensor(labels, output_shape, interp_method="nearest")

    # # map generation labels to segmentation values
    # labels = layers.ConvertLabels(
    #     generation_labels, dest_values=output_labels, name="labels_out"
    # )(labels)

    # build model (dummy layer enables to keep the labels when plugging this model to other models)
    image = KL.Lambda(lambda x: x[0], name="image_out")([image, labels])
    brain_model = Model(inputs=list_inputs, outputs=[image, labels])

    return brain_model


if __name__ == "__main__":
    for randomise_res_value in [True]:
        # TODO: replace this with a label image of your choice
        labels_dir = (
            "/om2/user/hgazula/SynthSeg/data/training_label_maps/training_seg_01.nii.gz"
        )

        labels_paths = utils.list_images_in_folder(labels_dir)
        subjects_prob = None
        labels_shape, aff, n_dims, _, header, atlas_res = utils.get_volume_info(
            labels_paths[0], aff_ref=np.eye(4)
        )

        n_channels = 1

        generation_labels, _ = utils.get_list_labels(labels_dir=labels_dir)
        output_labels = generation_labels
        n_neutral_labels = generation_labels.shape[0]
        target_res = None
        batchsize = 1
        flipping = True
        output_shape = None
        output_div_by_n = None

        prior_distributions = "uniform"

        generation_classes = np.arange(generation_labels.shape[0])
        prior_means = None
        prior_stds = None
        use_specific_stats_for_channel = False

        mix_prior_and_random = False

        scaling_bounds = 0.2
        rotation_bounds = 15
        shearing_bounds = 0.012
        translation_bounds = False

        nonlin_std = 4.0
        nonlin_scale = 0.04

        randomise_res = randomise_res_value
        print("randomise_res", randomise_res)

        max_res_iso = 4.0
        max_res_aniso = 8.0

        data_res = None
        thickness = None

        bias_field_std = 0.7
        bias_scale = 0.025
        return_gradients = False

        lab_to_im_model = labels_to_image_model(
            labels_shape=labels_shape,
            n_channels=n_channels,
            generation_labels=generation_labels,
            output_labels=output_labels,
            n_neutral_labels=n_neutral_labels,
            atlas_res=atlas_res,
            target_res=target_res,
            output_shape=output_shape,
            output_div_by_n=output_div_by_n,
            flipping=flipping,
            aff=np.eye(4),
            scaling_bounds=scaling_bounds,
            rotation_bounds=rotation_bounds,
            shearing_bounds=shearing_bounds,
            translation_bounds=translation_bounds,
            nonlin_std=nonlin_std,
            nonlin_scale=nonlin_scale,
            randomise_res=randomise_res,
            max_res_iso=max_res_iso,
            max_res_aniso=max_res_aniso,
            data_res=data_res,
            thickness=thickness,
            bias_field_std=bias_field_std,
            bias_scale=bias_scale,
            return_gradients=return_gradients,
        )
        out_shape = lab_to_im_model.output[0].get_shape().as_list()[1:]

        model_inputs_generator = build_model_inputs(
            path_label_maps=labels_paths,
            n_labels=len(generation_labels),
            batchsize=batchsize,
            n_channels=n_channels,
            subjects_prob=subjects_prob,
            generation_classes=generation_classes,
            prior_means=prior_means,
            prior_stds=prior_stds,
            prior_distributions=prior_distributions,
            use_specific_stats_for_channel=use_specific_stats_for_channel,
            mix_prior_and_random=mix_prior_and_random,
        )

        model_inputs = next(model_inputs_generator)
        [image, labels] = lab_to_im_model.predict(model_inputs)

        print(image.shape, labels.shape)
        print("Success")
