"""This script saves OpenNeuro MetaSearch volumes to HDF5."""
# Author: Jakub Kaczmarzyk <jakubk@mit.edu>

import numpy as np

import data_grabber
import utils


def load_metasearch():
    """Return pandas.DataFrame of MetaSearch data."""
    metasearch = data_grabber.MetaSearchJSON()

    # # The following two lines send JSON requests to MetaSearch's servers.
    # # This was done previously, and the JSON file was uploaded to DropBox.
    # metasearch.load_from_metasearch()
    # metasearch.add_mri_json_col()

    url = 'https://dl.dropbox.com/s/idfej9cazw4i0a5/metasearch_data.json'
    metasearch.load(url)
    metasearch.add_json_value('_id')
    metasearch.add_has_annotation()
    metasearch.add_annotation_url('aseg').add_annotation_url('brainmask')

    return metasearch.phenotype


def prepare_data(df):
    """Remove duplicates and subset entries that have brainmask."""
    df = df.set_index('_id').drop('_json', 1)
    df = df[~df.index.duplicated(keep='first')]
    return df.loc[df['has_brainmask'], :].copy()


def add_anat_mask(anat_url, mask_url, sinker, views, slice_shape):
    """Add slices of one anatomical and one brainmask image to the HDF5 file.

    Multiple views (i.e., 'axial', 'coronal', 'sagittal') can be supported,
    but the functionality is commented out.
    """
    anat = data_grabber.VolumeLoader(anat_url)
    mask = data_grabber.VolumeLoader(mask_url)

    anat.reoriented = utils.reorient_volume(anat.img)
    mask.reoriented = utils.reorient_volume(mask.img)

    for view in views:
        mask.rm_empty, anat.rm_empty = utils.remove_empty_slices(
            mask.reoriented.get_data(), anat.reoriented.get_data(), view)

        mask.resized = np.array([utils.resize_arr(s, slice_shape) for s in
                                 utils._gen_slices(mask.rm_empty, view)])

        anat.resized = np.array([utils.resize_arr(s, slice_shape) for s in
                                 utils._gen_slices(anat.rm_empty, view)])

        sinker.append(mask.resized, '/brainmask/{}'.format(view))
        sinker.append(anat.resized, '/anatomical/{}'.format(view))


def main(path, slice_shape, views, log_path, log_error_path):

    df = load_metasearch()
    df = prepare_data(df)

    datasets = ['/{}/{}'.format(g, v) for g in ['anatomical', 'brainmask']
                for v in views]

    logger = utils.get_logger('csv_output', log_path)
    logger_errors = utils.get_logger('errors', log_error_path)

    sinker = data_grabber.HdfSinker(path, datasets, slice_shape,
                                    compression='gzip', compression_opts=5)

    logger.info('n,id')

    for i, (index, entry) in enumerate(df.iterrows()):
        try:
            anat_url = entry['MRIs']
            mask_url = entry['url_brainmask']
            add_anat_mask(anat_url, mask_url, sinker, views, slice_shape)
            logger.info("{},{}".format(i, index))
        except Exception as e:
            logger_errors.error(str(index), exc_info=e)


if __name__ == '__main__':

    path = "/om/user/jakubk/neuro_nn/data/metasearch-anatomical-brainmask-slices.h5"
    slice_shape = (256, 256)  # Resize all slices to this shape.
    views = ['axial', 'coronal', 'sagittal']
    log_path = "/home/jakubk/neuro_nn/logs/log_axial.csv"
    log_error_path = "/home/jakubk/neuro_nn/logs/errors_axial.log"

    main(path, slice_shape, views, log_path, log_error_path)
