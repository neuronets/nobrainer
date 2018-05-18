.PHONY: all build clean-build clean-pyc clean install

PYTHON ?= python3

all: clean build

build: clean
	$(PYTHON) setup.py bdist_wheel

clean-build:
	rm -rf dist build

clean-pyc:
	find . -name "*.pyc" -type f -exec rm -f {} +
	find . -name "__pycache__" -type d -exec rm -rf {} +

clean: clean-build clean-pyc

install:
	$(PYTHON) setup.py install
