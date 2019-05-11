#!/bin/bash

source venv/bin/activate
export FLASK_APP=diab.py
python3 -m flask run
deactivate 

