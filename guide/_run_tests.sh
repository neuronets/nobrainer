#!/usr/bin/env bash
#
# Test the notebooks in this directory.

set -e

# Convert notebooks to .py files.
jupyter-nbconvert --to python *.ipynb

# Remove anything from `model.fit` and below in scripts.
# Model fitting may crash travis.
# Also remove references to `pip install`.
sed -i -e '/model.fit/,$d; /pip install/d;' *.py

# Add a print statement to the bottom of each script so we know it finished.
for f in *.py; do echo print\(\"++ FINISHED $f ++\"\) >> $f; done

# Run!
for f in *.py; do jupyter-run $f || exit 1; done
