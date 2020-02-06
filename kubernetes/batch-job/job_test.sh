#!/bin/bash

# persistent configs
NOTEBOOK_ROOT="/sbksvol/misc_tests/"
NOTEBOOK_NAME="hello_world.ipynb"
# CONFIG_PATH="${NOTEBOOK_ROOT}config_files/config_model-$1"

# convert jupyter notebook to python script
jupyter nbconvert --to python "${NOTEBOOK_ROOT}${NOTEBOOK_NAME}"

# run the generated script with ipython
ipython "${NOTEBOOK_ROOT}${NOTEBOOK_NAME%.*}.py" #$CONFIG_PATH
