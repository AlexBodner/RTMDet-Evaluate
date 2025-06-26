#!/bin/bash
set -e

current_path=$(pwd)
#!/bin/bash
set -e

current_path=$(pwd)


# Create virtual environment using virtualenv
virtualenv .venv
project_dir="$current_path"

# Define full paths
venv_dir="$project_dir/.venv"

VENV_PY="$venv_dir/bin/python"
VENV_PIP="$venv_dir/bin/pip"
VENV_MIM="$venv_dir/bin/mim"  

# Upgrade pip
$VENV_PY -m pip install --upgrade pip

# Install dependencies
$VENV_PIP install -r requirements.txt

$VENV_PIP install -U openmim

$VENV_PY -m mim install mmcv==2.0.0
# Run script
export MPLBACKEND=Agg
export
$VENV_PY main.py
  

done


