#!/bin/bash

# Just link to 'en' because the spacy model is loaded as a dataset 
python -m spacy link en_core_web_sm en
# python -m spacy download en