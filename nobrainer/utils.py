"""Utilities for Nobrainer."""

from collections import namedtuple
import csv
import os
import tempfile

import numpy as np
import psutil
import tensorflow as tf

_cache_dir = os.path.join(tempfile.gettempdir(), "nobrainer-data")


def get_data(cache_dir=_cache_dir):
    """Download sample features and labels. The features are T1-weighted MGZ
    files, and the labels are the corresponding aparc+aseg MGZ files, created
    with FreeSurfer. This will download 46 megabytes of data.

    These data can be found at
    https://datasets.datalad.org/workshops/nih-2017/ds000114/.

    Parameters
    ----------
    cache_dir: str, directory where to save the data. By default, saves to a
        temporary directory.

    Returns
    -------
    List of `(features, labels)`.
    """

    os.makedirs(cache_dir, exist_ok=True)
    URLHashPair = namedtuple("URLHashPair", "sub x_hash y_hash")
    hashes = [
        URLHashPair(
            sub="sub-01",
            x_hash="67d0053f021d1d137bc99715e4e3ebb763364c8ce04311b1032d4253fc149f52",
            y_hash="7a85b628653f24e2b71cbef6dda86ab24a1743c5f6dbd996bdde258414e780b5",
        ),
        URLHashPair(
            sub="sub-02",
            x_hash="c0fee669a34bf3b43c8e4aecc88204512ef4e83f2e414640a5abc076b435990c",
            y_hash="c92357c2571da72d15332b2b4838b94d442d4abd3dbddc4b54202d68f0e19380",
        ),
        URLHashPair(
            sub="sub-03",
            x_hash="e2bba954e37f5791260f0ec573456e3293bbd40dba139bb1af417eaaeabe63e6",
            y_hash="e9204f0d50f06a89dd1870911f7ef5e9808e222227799a5384dceeb941ee8f9d",
        ),
        URLHashPair(
            sub="sub-04",
            x_hash="deec5245a2a5948f7e1053ace8d8a31396b14a96d520c6a52305434e75abe1e8",
            y_hash="c50e33a3f87aca351414e729b7c25404af364dfe5dd1de5fe380a460cbe9f891",
        ),
        URLHashPair(
            sub="sub-05",
            x_hash="8a7fe84918f3f80b87903a1e8f7bd20792c0ebc7528fb98513be373258dfd6c0",
            y_hash="682f52633633551d6fda71ede65aa41e16c332ebf42b4df042bc312200b0337c",
        ),
        URLHashPair(
            sub="sub-06",
            x_hash="f9a0c40bcd62d7b7e88015867ab5d926009b097ac3235499a541ac9072dd90c8",
            y_hash="31c842969af9ac178361fa8c13f656a47d27d95357abaf3e7f3521671aa17929",
        ),
        URLHashPair(
            sub="sub-07",
            x_hash="9de3b7392f5383e7391c5fcd9266d6b7ab6b57bc7ab203cc9ad2a29a2d31a85b",
            y_hash="b2e48bbfc4185261785643fc8ab066be5f97215b5a9b029ade1ffb12d54d616e",
        ),
        URLHashPair(
            sub="sub-08",
            x_hash="361098fc69c280970bb0b0d7ea6aba80d383c12e3ccfe5899693bc35b68efbe4",
            y_hash="0c980ef851b1391f580d91fc87c10d6d30315527cc0749c1010f2b7d5819a009",
        ),
        URLHashPair(
            sub="sub-09",
            x_hash="1456b35112297df5caacb9d33cb047aa85a3a5b4db3b4b5f9a5c2e189a684e1a",
            y_hash="696f1e9fef512193b71580292e0edc5835f396d2c8d63909c13668ef7bed433b",
        ),
        URLHashPair(
            sub="sub-10",
            x_hash="97447f17402e0f9990cd0917f281704893b52a9b61a3241b23a112a0a143d26e",
            y_hash="97a7947ba1a28963714c9f5c82520d9ef803d005695a0b4109d5a73d7e8a537b",
        ),
    ]
    x_filename = "t1.mgz"
    y_filename = "aparc+aseg.mgz"
    url_template = (
        "https://datasets.datalad.org/workshops/nih-2017/ds000114/derivatives/"
        "freesurfer/{sub}/mri/{fname}"
    )
    output = [("features", "labels")]
    for h in hashes:
        x_origin = url_template.format(sub=h.sub, fname=x_filename)
        y_origin = url_template.format(sub=h.sub, fname=y_filename)
        x_fname = h.sub + "_" + x_origin.rsplit("/", 1)[-1]
        y_fname = h.sub + "_" + y_origin.rsplit("/", 1)[-1]
        x_out = tf.keras.utils.get_file(
            fname=x_fname, origin=x_origin, file_hash=h.x_hash, cache_dir=cache_dir
        )
        y_out = tf.keras.utils.get_file(
            fname=y_fname, origin=y_origin, file_hash=h.y_hash, cache_dir=cache_dir
        )
        output.append((x_out, y_out))

    csvpath = os.path.join(cache_dir, "filepaths.csv")
    with open(csvpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output)

    return csvpath


class StreamingStats:
    """Object to calculate statistics on streaming data.

    Compatible with scalars and n-dimensional arrays.

    Examples
    --------

    ```python
    >>> s = StreamingStats()
    >>> s.update(10).update(20)
    >>> s.mean()
    15.0
    ```

    ```python
    >>> import numpy as np
    >>> a = np.array([[0, 2], [4, 8]])
    >>> b = np.array([[2, 4], [8, 16]])
    >>> s = StreamingStats()
    >>> s.update(a).update(b)
    >>> s.mean()
    array([[ 1.,  3.],
       [ 6., 12.]])
    ```
    """

    def __init__(self):
        self._n_samples = 0
        self._current_mean = 0.0
        self._M = 0.0

    def update(self, value):
        """Update the statistics with the next value.

        Parameters
        ----------
        value: scalar, array-like

        Returns
        -------
        Modified instance.
        """
        if self._n_samples == 0:
            self._current_mean = value
        else:
            prev_mean = self._current_mean
            curr_mean = prev_mean + (value - prev_mean) / (self._n_samples + 1)
            _M = self._M + (prev_mean - value) * (curr_mean - value)
            # Set the instance attributes after computation in case there are
            # errors during computation.
            self._current_mean = curr_mean
            self._M = _M
        self._n_samples += 1
        return self

    def mean(self):
        """Return current mean of streaming data."""
        return self._current_mean

    def var(self):
        """Return current variance of streaming data."""
        return self._M / self._n_samples

    def std(self):
        """Return current standard deviation of streaming data."""
        return self.var() ** 0.5

    def entropy(self):
        """Return current entropy of streaming data."""
        eps = 1e-07
        mult = np.multiply(np.log(self.mean() + eps), self.mean())
        return -mult
        # return -np.sum(mult, axis=axis)


def get_num_parallel():
    # Get number of processes allocated to the current process.
    # Note the difference from `os.cpu_count()`.
    try:
        num_parallel_calls = len(psutil.Process().cpu_affinity())
    except AttributeError:
        num_parallel_calls = psutil.cpu_count()
    return num_parallel_calls
