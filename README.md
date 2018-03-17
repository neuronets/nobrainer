# nobrainer

Neural network to identify brains in structural magnetic resonance images. This is a work in progress.


## Get the container

```shell
$ docker pull kaczmarj/nobrainer
# or
$ singularity build nobrainer.simg docker://kaczmarj/nobrainer
```


## Train your own models

Models can be trained on neuroimaging volumes or HDF5 data.

`$` in the examples below indicate command-line calls. All of the commands below can be run within the Nobrainer container.

### Train on neuroimaging volumes

Volumes must all be the same shape. No preprocessing is necessary.

```shell
$ ./train_on_volumes.py -n2 -mhighres3dnet \
  -oAdam -l0.001 -b6 \
  --brainmask \
  --vol-shape 256 256 256 \
  --block-shape 64 64 64  \
  --strides 32 32 32 \
  --n-epochs=5 \
  --model-dir path/to/model \
  --csv features_labels.csv \
  --eval-csv evaluation_data.csv
```


### Convert volumes to HDF5

Note: all volumes must have the same shape.

```shell
$ ./vols2hdf5.py --help
```

#### Examples:

##### Convert pairs of T1 and aparc+aseg files from a FreeSurfer SUBJECTS_DIR to HDF5:

This also z-scores the feature data per volume.

```shell
$ ./vols2hdf5.py -o output.h5 \
  --block-shape 128 128 128 \
  --block-shape 64 64 64 \
  -fdt float32 -ldt int32 \
  --chunksize 75 --ncpu 6 \
  --normalize zscore \
  --compression gzip --compression-opts 1 \
  $SUBJECTS_DIR
```

The resulting HDF5 file has the structure:

```
/
├── 128x128x128
│   ├── features
│   └── labels
└── 64x64x64
    ├── features
    └── labels
```

##### Convert features and labels volumes using filepaths stored in CSV to HDF5:

```shell
$ ./vols2hdf5.py -o output.h5 \
  --block-shape 128 128 128 \
  --block-shape 64 64 64 \
  -fdt float32 -ldt int32 \
  --chunksize 75 --ncpu 6 \
  --compression gzip --compression-opts 1 \
  files.csv
```

The resulting HDF5 file has the structure:

```
/
├── 128x128x128
│   ├── features
│   └── labels
└── 64x64x64
    ├── features
    └── labels
```

### Train on HDF5 data

Models can be trained with a command-line interface. For more granular control, write a Python train script (refer to the `train` function in `train_on_hdf5.py`).

```shell
$ ./train_on_hdf5.py --help
```

#### Train a two-class MeshNet model that classifies brain/not-brain.

The `--brainmask` option binarizes the labels.

```shell
$ ./train_on_hdf5.py  --n-classes=2 --model=meshnet \
  --model-dir=path/to/checkpoints \
  --optimizer=Adam --learning-rate=0.001 --batch-size=1 \
  --hdf5path data.h5 \
  --xdset=/128x128x128/features --ydset=/128x128x128/labels \
  --block-shape 128 128 128 \
  --brainmask
```

#### Train a multi-class HighRes3DNet model to classify multiple brain structures.

The `--aparcaseg-mapping` option is used to convert the label values to continuous labels beginning at 0.

```shell
$ ./train_on_hdf5.py  --n-classes=7 --model=highres3dnet \
  --model-dir=path/to/checkpoints \
  --optimizer=Adam --learning-rate=0.001 --batch-size=1 \
  --hdf5path data.h5 \
  --xdset=/128x128x128/features --ydset=/128x128x128/labels \
  --block-shape 128 128 128 \
  --aparcaseg-mapping=mapping.csv
```

The `mapping.csv` file looks like this:

```
original,new,label
0,0,Unknown
2,1,Left-Cerebral-White-Matter
7,2,Left-Cerebellum-White-Matter
8,3,Left-Cerebellum-Cortex
41,4,Right-Cerebral-White-Matter
46,5,Right-Cerebellum-White-Matter
47,6,Right-Cerebellum-Cortex
```


#### Train a two-class HighRes3DNet model to classify hippocampus/not-hippocampus.

```shell
$ ./train_on_hdf5.py  --n-classes=2 --model=highres3dnet \
  --model-dir=path/to/checkpoints \
  --optimizer=Adam --learning-rate=0.001 --batch-size=1 \
  --hdf5path data.h5 \
  --xdset=/128x128x128/features --ydset=/128x128x128/labels \
  --block-shape 128 128 128 \
  --aparcaseg-mapping=mapping.csv
```

The `mapping.csv` file looks like this:

```
original,new,label
0,0,unknown
17,1,left-hippocampus
53,1,right-hippocampus
```
