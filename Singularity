# Singularity bootstrap definition file for the brain extraction neural network
# project.
#
# The vast majority of this file is copied from Satra's keras-gpu Singularity
# file: https://singularity-hub.org/containers/849/
#
# It would be nice to bootstrap Satra's Singularity image, but this is not
# possible right now. See
# https://github.com/singularityware/singularity/issues/667.


# To modify this file for your own system, use nvidia-smi
# to check your driver version and adjust the variable
# NV_DRIVER_VERSION below.
# Check sections <---- EDIT:

BootStrap: docker
From: tensorflow/tensorflow:1.1.0-rc1-gpu-py3

%runscript
    # When executed, the container will run Python with the TensorFlow module
    exec /usr/bin/python "$@"

%post
    # Set up some required environment defaults
    export LC_ALL=C
    export PATH=/bin:/sbin:/usr/bin:/usr/sbin:$PATH

    # add universe repo and install some packages
    sed -i '/xenial.*universe/s/^#//g' /etc/apt/sources.list
    locale-gen en_US.UTF-8

    echo "Installing Python packages ..."
    pip install --no-cache-dir -U pip
    pip install --no-cache-dir h5py keras nibabel

    NV_DRIVER_VERSION=375.20      # <---- EDIT: CHANGE THIS FOR YOUR SYSTEM
    NV_DRIVER_FILE=NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION}.run

    working_dir=$(pwd)
    # download and run NIH NVIDIA driver installer
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/${NV_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION}.run

    echo "Unpacking NVIDIA driver into container..."
    cd /usr/local/
    sh ${working_dir}/${NV_DRIVER_FILE} -x
    rm ${working_dir}/${NV_DRIVER_FILE}
    mv NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION} NVIDIA-Linux-x86_64
    cd NVIDIA-Linux-x86_64/
    for n in *.$NV_DRIVER_VERSION; do
        ln -v -s $n ${n%.375.20}   # <---- EDIT: CHANGE THIS IF DRIVER VERSION
    done
    ln -v -s libnvidia-ml.so.$NV_DRIVER_VERSION libnvidia-ml.so.1
    ln -v -s libcuda.so.$NV_DRIVER_VERSION libcuda.so.1
    cd $working_dir

    echo "Adding NVIDIA PATHs to /environment..."
    NV_DRIVER_PATH=/usr/local/NVIDIA-Linux-x86_64
    echo "

LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$NV_DRIVER_PATH:\$LD_LIBRARY_PATH
PATH=$NV_DRIVER_PATH:\$PATH
export PATH LD_LIBRARY_PATH

" >> /environment

%test
    # Ensure that TensorFlow can be imported
    /usr/bin/python -c "import tensorflow as tf"
    # Ensure that keras can be imported
    /usr/bin/python -c "from keras.models import Sequential"
    # Ensure that nibabel can be imported
    /usr/bin/python -c "import nibabel as nb"
    # Ensure that h5py can be imported.
    /usr/bin/python -c "import h5py"
