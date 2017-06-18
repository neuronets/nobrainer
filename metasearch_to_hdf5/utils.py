"""Utility functions."""
# Author: Jakub Kaczmarzyk <jakubk@mit.edu>


def _find_dict_in_list(list_of_dict, key, val):
    """Given a list of dictionaries, return the first dictionary that has the
    given key:value pair.
    """
    try:
        for d in list_of_dict:
            if (key, val) in d.items():
                return d
    except TypeError:
        pass


def _gen_slices(arr, view):
    """Generate slices in a certain `view`, where `view` can be 'axial',
    'coronal', or 'sagittal'.
    """
    view_dims = {'axial': 2,
                 'coronal': 1,
                 'sagittal': 0}

    for i in range(arr.shape[view_dims[view]]):
        if view == 'axial':
            yield arr[:, :, i]
        elif view == 'coronal':
            yield arr[:, i, :]
        elif view == 'sagittal':
            yield arr[i, :, :]


def get_logger(name, path):
    """Return logger that writes to file `path`."""
    import logging
    logger = logging.getLogger(name)
    handler = logging.FileHandler(path)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _get_suffix(headers, url):
    """Return suffix of downloaded file either from header of URL (either
    .nii.gz or .mgz).
    """
    # This try-except block is necessary in cases where the extension is not
    # present in the URL. Ideally, the file type would be determined by
    # the content and not by the filename.
    try:
        if '.nii.gz' in headers['Content-Disposition']:
            suffix = ".nii.gz"
        elif '.mgz' in headers['Content-Disposition']:
            suffix = ".mgz"
        return suffix
    except Exception:
        pass

    if url.endswith('.nii.gz'):
        suffix = '.nii.gz'
    elif url.endswith('.mgz'):
        suffix = '.mgz'
    else:
        raise ValueError("URL extension not .nii.gz or .mgz.")
    return suffix


def load_volume_from_url(url, **kwargs):
    """Return image object and image data from URL. Kwargs are for
    `nibabel.load()`.
    """
    import tempfile
    import nibabel as nib
    import requests

    response = requests.get(url)
    response.raise_for_status()
    suffix = _get_suffix(response.headers, url)

    with tempfile.NamedTemporaryFile(suffix=suffix) as fp:
        fp.write(response.content)
        img = nib.load(fp.name, **kwargs)
        return img, img.get_data()


def remove_empty_slices(mask_arr, other_arr, view):
    """Return `mask_arr` and `other_arr` with empty `view` slices in `mask_arr`
    removed. `view` can be 'axial', 'coronal', or 'sagittal'.
    """
    if view == 'axial':
        _mask = mask_arr.any(axis=(0, 1))
        return mask_arr[:, :, _mask], other_arr[:, :, _mask]
    elif view == 'coronal':
        _mask = mask_arr.any(axis=(0, 2))
        return mask_arr[:, _mask, :], other_arr[:, _mask, :]
    elif view == 'sagittal':
        _mask = mask_arr.any(axis=(1, 2))
        return mask_arr[_mask, :, :], other_arr[_mask, :, :]
    else:
        raise ValueError("view not understood: {}".format(view))


def reorient_volume(img):
    """Attempt to reorient img to RAS+."""
    import nibabel as nib
    return nib.as_closest_canonical(img)


def resize_arr(arr, new_shape, **kwargs):
    """Return array resized to `new_shape`."""
    from scipy.misc import imresize  # Requires Pillow or PIL.

    return imresize(arr, new_shape, **kwargs)
