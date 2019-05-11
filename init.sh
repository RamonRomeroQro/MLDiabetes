#!/bin/bash
echo "creating venv"
python3 -m venv venv
source venv/bin/activate
echo "install dependencies"
pip3 install --upgrade pip
pip3 install -r req.txt 
deactivate
