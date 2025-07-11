#!/bin/bash
# setup.sh

eval "$(conda shell.bash hook)"

# Create the environment
conda env create -f environment.yml
# Check if creation succeeded
if [ $? -eq 0 ]; then
  echo "Environment created successfully. Running make..."
  cd dtw
  make
  cd ..
else
  echo "Failed to create conda environment."
  exit 1
fi

# Activate the environment
conda activate child-speech-bench
echo "Activated conda environment: child-speech-bench"

