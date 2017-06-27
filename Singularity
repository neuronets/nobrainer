# Singularity bootstrap definition file for the brain extraction neural network
# project.

BootStrap: shub
From: satra/om-images:keras-gpu

%runscript
    exec /usr/bin/python "$@"

%post
    echo "Installing Python packages ..."
    pip install --no-cache-dir -U pip
    pip install --no-cache-dir h5py

%test
    # Ensure that some of the Python packages can be imported.
    /usr/bin/python -c "import tensorflow as tf"
    /usr/bin/python -c "from keras.models import Sequential"
    /usr/bin/python -c "import nibabel as nb"
    /usr/bin/python -c "import h5py"
