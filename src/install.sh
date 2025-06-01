#!/bin/bash
####################################################################################################

# Exit immediately if a command exits with a non-zero status
set -e

# Create a virtual environment named 'venv'
python3 -m venv venv

# Upgrade pip within the 'venv'
venv/bin/pip install --upgrade pip

# Install each required package using the pip from the virtual environment
venv/bin/pip install pandas
venv/bin/pip install numpy
venv/bin/pip install scikit-learn
venv/bin/pip install pecanpy
venv/bin/pip install matplotlib

echo "All dependencies have been successfully installed in the 'venv' virtual environment"

# To start using the virtual environment, run: activat (alias: source venv/bin/activate)


####################################################################################################


