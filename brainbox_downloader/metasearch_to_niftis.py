"""This script saves OpenNeuro MetaSearch volumes to NIFTI files."""
# Author: Jakub Kaczmarzyk <jakubk@mit.edu>

import os

import utils

BASE_PATH = os.path.join(os.sep, 'storage', 'gablab001', 'data', 'nobrainer',
                         'volumes')

log_base = os.path.join(os.sep, 'home', 'jakubk', 'nobrainer')
logger_stdout = utils.get_logger('logger-out', os.path.join(log_base, 'out.log'))
logger_stderr = utils.get_logger('logger-err', os.path.join(log_base, 'err.log'))


def load_data():
    """Return long pd.DataFrame of anatomical and brainmask volumes."""
    import pandas as pd

    url = "https://dl.dropbox.com/s/68apfyeg2ncdmb7/metasearch_data_for_niftis_pruned.csv"
    return pd.read_csv(url)


def save_response(filepath, bytes_):
    with open(filepath, 'wb') as fp:
        fp.write(bytes_)


def _download_one_volume(_id, scan_type, url):
    import requests
    import pandas as pd
    import utils

    if pd.isnull(url):
        return False

    response = requests.get(url)
    response.raise_for_status()

    try:
        suffix = utils._get_suffix(response.headers, url)
    except ValueError:  # suffix not understood
        suffix = ""  # save without suffix, we can change later

    filepath = "{}_{}{}".format(_id, scan_type, suffix)
    filepath = os.path.join(BASE_PATH, scan_type, filepath)

    save_response(filepath, response.content)

    return True


def _download_one_volume_wrapped(row):
    # _id, scan_type, url = row
    _id, scan_type, url = row['_id'], row['variable'], row['url']
    print(_id, scan_type)
    message = "{} {}".format(_id, scan_type)
    try:
        _download_one_volume(_id, scan_type, url)
        logger_stdout.info(message)
    except Exception as e:
        message += "\n" + str(e) + "\n\n"
        logger_stderr.error(message)


def download_volumes(df):
    df.apply(_download_one_volume_wrapped, axis=1)


def apply_parallel(df, func):
    from multiprocessing import Pool, cpu_count
    import numpy as np

    n_cpus = cpu_count()
    list_of_dfs = np.array_split(df, n_cpus)
    with Pool(n_cpus) as p:
        p.map(func, list_of_dfs)
    return True


def main():
    df_niftis = load_data()
    apply_parallel(df_niftis, download_volumes)


if __name__ == '__main__':
    main()
