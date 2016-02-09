#!/usr/bin/env bash

find . -name '*.pyc' -delete
find . -name "*.egg-info" -type d -exec rm -r "{}" \;  2> /dev/null
find . -name "*__pycache__" -type d -exec rm -r "{}" \;  2> /dev/null

pip install -e .

py.test --cache-clear

find . -name '*.pyc' -delete
find . -name "*.egg-info" -type d -exec rm -r "{}" \;  2> /dev/null
find . -name "*__pycache__" -type d -exec rm -r "{}" \;  2> /dev/null
