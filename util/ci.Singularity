Bootstrap: docker
From: tensorflow/tensorflow:latest-gpu

# Reads:
#   requirements.txt
#
# Creates:
#   venv
#
# Usage:
#   $ ./venv.sif python myscript.py
#   runs myscript.py inside the virtual environment defined by requirements.txt
#
#   $ ./venv.sif python
#   opens python shell inside the virtual environment defined by requirements.txt

%post
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt install -y python3 python3-venv
  apt-get clean

  ln -sf /usr/bin/python3 /usr/bin/python

  # default sh on ubuntu is dash
  ln -sf /bin/bash /bin/sh

%environment
  export LC_ALL=C

%runscript
  if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
  else
    source venv/bin/activate
    if [ "requirements.txt" -nt "venv" ]; then
      echo "requirements.txt is newer than venv: updating venv"
      python3 -m pip install -r requirements.txt
      touch venv
    fi
  fi

  $@
