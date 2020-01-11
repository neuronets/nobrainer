#!/usr/bin/env bash
#
# Test the notebooks in this directory.

set -e

# Convert notebooks to .py files.
jupyter-nbconvert --to python *.ipynb

# Remove anything from `model.fit` and below in scripts.
# Model fitting may crash travis.
# Also remove references to `pip install`.
sed -i -e '/model.fit/,$d; /tf.distribute./,$d; /TPU_WORKER/,$d; /pip install/d;' *.py

# Run!
for f in *.py
do
  (echo "++ STARTING $f" && ipython $f && echo "++ FINISHED $f") \
  || (echo "!! ERROR on $f" && exit 1)
done
