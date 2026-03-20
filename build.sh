#!/usr/bin/env bash

pip install --upgrade pip
pip install --upgrade setuptools wheel

# force binary install (no source build)
pip install --only-binary=:all: -r requirements.txt