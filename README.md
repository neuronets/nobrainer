# nobrainer

Neural network to identify brains in structural magnetic resonance images. This is a work in progress.


## Get the container

```shell
$ docker pull kaczmarj/nobrainer
# or
$ singularity build nobrainer.simg docker://kaczmarj/nobrainer
```


## Train your own models

### Convert volumes to HDF5

Note: all volumes must have the same shape.

```shell
$ ./vols2hdf5.py --help
```

Examples:

- Convert pairs of T1 and aparc+aseg files from a FreeSurfer SUBJECTS_DIR to HDF5:

```shell
$ ./vols2hdf5.py -o output.h5 \
  --block-shape 128 128 128 \
  --block-shape 64 64 64 \
  -fdt float32 -ldt int32 \
  --chunksize 75 --ncpu 6 \
  --compression gzip --compression-opts 1 \
  $SUBJECTS_DIR
```

- Convert features and labels volumes using filepaths stored in CSV to HDF5:

```shell
$ ./vols2hdf5.py -o output.h5 \
  --block-shape 128 128 128 \
  --block-shape 64 64 64 \
  -fdt float32 -ldt int32 \
  --chunksize 75 --ncpu 6 \
  --compression gzip --compression-opts 1 \
  files.csv
```
