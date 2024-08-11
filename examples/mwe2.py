import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras.layers as KL
import numpy as np
import tensorflow as tf
from keras.models import Model

from nobrainer.ext.lab2im import edit_tensors as l2i_et
from nobrainer.ext.lab2im import layers, utils
from nobrainer.ext.lab2im.edit_volumes import get_ras_axes
from nobrainer.ext.SynthSeg.model_inputs import build_model_inputs
from nobrainer.models.labels_to_image_model import get_shapes


def sample_model(
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
    atlas_res = atlas_res[0]

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

    max_res_iso = np.array(
        utils.reformat_to_list(max_res_iso, length=n_dims, dtype="float")
    )
    max_res_aniso = np.array(
        utils.reformat_to_list(max_res_aniso, length=n_dims, dtype="float")
    )
    output1 = layers.SampleResolution(atlas_res, max_res_iso, max_res_aniso)(
        means_input
    )

    brain_model = Model(inputs=list_inputs, outputs=output1)
    return brain_model


if __name__ == "__main__":
    for randomise_res_value in [True]:
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

        sam_mod = sample_model(
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
        )

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
        print("start prediction")
        # output = lab_to_im_model(model_inputs)
        # sam_mod.summary()
        output = sam_mod.predict(model_inputs)

        # print(image.shape, labels.shape)
        print("Success")
