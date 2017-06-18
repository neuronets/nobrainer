"""Objects and methods to grab data from MetaSearch and BrainBox."""
# Author: Jakub Kaczmarzyk <jakubk@mit.edu>

from urllib.parse import urljoin

import h5py
import pandas as pd

import utils


class MetaSearchJSON:
    """Object to work with phenotype data on MetaSearch.

    Attributes
    ----------
    _json_col : column name for JSON responses.
    brainbox_root : root BrainBox URL.
    phenotype : pd.DataFrame of MetaSearch phenotype data.
    """
    def __init__(self):
        self.brainbox_root = 'http://brainbox.pasteur.fr'
        self.phenotype = None
        self._json_col = '_json'

    def load_from_metasearch(self, **kwargs):
        """Load pd.DataFrame of all MetaSearch data, using the CSV file on
        OpenNeuro's servers.
        """
        phenotype_url = 'http://openneu.ro/metasearch/data/phenotype_mri.csv'
        self.phenotype = pd.read_csv(phenotype_url)
        return self

    def load(self, path, **kwargs):
        """Load pd.DataFrame from JSON."""
        self.phenotype = pd.read_json(path, **kwargs)
        return self

    def save(self, path, **kwargs):
        """Save pd.DataFrame to JSON."""
        self.phenotype.to_json(path, **kwargs)
        return self

    def _get_json_request(self, mri_url, **kwargs):
        """Return dictionary with information about a BrainBox MR image, given
        a URL that points to the MR image. Kwargs are passed to the JSON
        request.
        """
        import requests

        params = {'url': mri_url, **kwargs}
        json_url = urljoin(self.brainbox_root, '/mri/json')
        return requests.get(json_url, params=params).json()

    def add_mri_json_col(self, mri_url_col='MRIs', **kwargs):
        """Add column to DataFrame with JSON responses for each MR image.

        Note: this can take more than 30 minutes.
        """
        self.phenotype.loc[:, self._json_col] = \
            self.phenotype.loc[:, mri_url_col].apply(self._get_json_request,
                                                     **kwargs)
        return self

    @staticmethod
    def _get_annotation_names(json_response):
        """Return list of annotation names from JSON response. If no names are
        found, return empty list."""
        names = []
        try:
            for annotation in json_response['mri']['atlas']:
                try:
                    names.append(annotation['name'])
                except KeyError:
                    pass
        except TypeError:
            pass
        return names

    def get_unique_annotation_names(self):
        """Return set of unique annotation names in MetaSearch data."""
        tmp = (self.phenotype.loc[:, self._json_col]
               .apply(self._get_annotation_names))
        return set([item for sublist in tmp.tolist() for item in sublist])

    @staticmethod
    def _has_annotation(json_request, name):
        """Return True if json_request contains an annotation with
        name==`name`. Else, return False."""
        try:
            for annotation in json_request['mri']['atlas']:
                try:
                    if annotation['name'] == name:
                        return True
                except KeyError:
                    pass
        except TypeError:
            pass
        return False

    def add_has_annotation(self, annotation_names=None):
        """Add a column for each annotation name in `annotation_names`
        indicating whether that MetaSearch entry has an annotation with that
        name. By default, annotation_names is a list of the unique annotation
        names in self.phenotype.
        """
        if annotation_names is None:
            annotation_names = self.get_unique_annotation_names()
        for name in annotation_names:
            col = "has_{}".format(name)
            self.phenotype.loc[:, col] = (self.phenotype.loc[:, self._json_col]
                                          .apply(self._has_annotation,
                                                 name=name))
        return self

    def _get_annotation_file_url(self, row, annotation_name, key='filename'):
        """Return the value corresponding to `key` in the dictionary associated
        with `annotation_name`, where `row` is a row (pd.Series) in the
        phenotype data.
        """
        try:
            filename = \
                utils._find_dict_in_list(row[self._json_col]['mri']['atlas'],
                                        'name', annotation_name)[key]
        except TypeError:
            return None
        path = urljoin(row[self._json_col]['url'], filename)
        return urljoin(self.brainbox_root, path)

    def add_annotation_url(self, annotation_name):
        """Add a column with the file url for annotation_name, where the new
        column name is 'url_<annotation_name>'.
        """
        colname = "url_{}".format(annotation_name)
        self.phenotype.loc[:, colname] = \
            self.phenotype.apply(self._get_annotation_file_url, axis=1,
                                 annotation_name=annotation_name)
        return self

    def add_json_value(self, key):
        """Add a column with the value in `key` in the JSON response per entry,
        where the column name is `key`.
        """
        def _get_value(json_response, key):
            try:
                return json_response[key]
            except (KeyError, TypeError):
                return None
        self.phenotype.loc[:, key] = \
            self.phenotype.loc[:, self._json_col].apply(_get_value, key=key)
        return self


class VolumeLoader:
    """Object to load volume from URL."""
    def __init__(self, url):
        self.url = url
        self.img, self.data = self._load(url)

    @staticmethod
    def _load(url):
        return utils.load_volume_from_url(url)


class HdfSinker:
    """Object to initialize and append data to HDF5 file. Kwargs are for
    h5py dataset creation.
    """
    def __init__(self, path, datasets, slice_shape, overwrite=False, **kwargs):
        self.path = path
        self.datasets = datasets
        self.slice_shape = slice_shape

        if not overwrite:
            import os
            if os.path.isfile(self.path):
                raise FileExistsError("File already exists. Use overwrite=True "
                                      "to overwrite.")

        with h5py.File(self.path, 'w') as fp:
            for d in datasets:
                fp.create_dataset(d, dtype='i8', shape=(0, *self.slice_shape),
                                  maxshape=(None, *self.slice_shape), **kwargs)

    def append(self, data, dataset):
        """Append data to HDF5 dataset, and return indices of appended items.
        """
        # https://stackoverflow.com/a/25656175/5666087

        with h5py.File(self.path, 'a') as fp:
            original_len = fp[dataset].shape[0]
            n_new_items = original_len + data.shape[0]
            fp[dataset].resize(n_new_items, axis=0)
            fp[dataset][-data.shape[0]:] = data
            return (original_len, n_new_items)
