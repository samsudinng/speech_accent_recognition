#!/bin/sh/

conda create -n $1 python=3.7
source activate $1
python -m venv $1
source $1/bin/activate
python -m pip install --upgrade pip==20.3
pip install -r requirements.txt
