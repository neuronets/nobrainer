```
conda create -n synthfix python=3.10 pip
git clone git@github.com:neuronets/nobrainer.git
cd nobrainer
pip install -e .
git checkout synthseg_fix
# please check the TODO in examples/mwe.py
python examples/mwe.py
```
