#!/bin/bash
pip install tensorboardX==1.4
pip install fastai==1.0.11
# svn export https://github.com/fastai/fastai.git/trunk/courses/dl2/imdb_scripts
python -m spacy download en
