#!/bin/bash
####################################################################################################
# Exit immediately if a command exits with a non-zero status
set -e

# Define conda environment
ENV_NAME="cenv"
PYTHON_VERSION="3.13.3"

# Check if conda environment already exists; if not, create it
if conda info --envs | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Conda environment ${ENV_NAME} already exists."
else
    echo "Creating conda environment ${ENV_NAME} with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Source conda's initialization script and activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Define ANSI color codes
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m"  # No Color

echo -e "\n${YELLOW}Installing pandas...${NC}"
conda install pandas -y -c conda-forge
echo -e "\n${GREEN}Pandas installation complete${NC}"

echo -e "\n${YELLOW}Installing numpy...${NC}"
conda install numpy -y -c conda-forge
echo -e "\n${GREEN}Numpy installation complete${NC}"

echo -e "\n${YELLOW}Installing scikit-learn...${NC}"
conda install scikit-learn -y -c conda-forge
echo -e "\n${GREEN}Scikit-learn installation complete${NC}"

echo -e "\n${YELLOW}Installing scipy...${NC}"
conda install scipy -y -c conda-forge
echo -e "\n${GREEN}Scipy installation complete${NC}"

echo -e "\n${YELLOW}Installing matplotlib...${NC}"
conda install matplotlib -y -c conda-forge
echo -e "\n${GREEN}Matplotlib installation complete${NC}"

echo -e "\n${YELLOW}Installing UMAP (umap-learn)...${NC}"
conda install umap-learn -y -c conda-forge
echo -e "\n${GREEN}UMAP installation complete${NC}"

echo -e "\n${YELLOW}Installing pyarrow...${NC}"
conda install pyarrow -y -c conda-forge
echo -e "\n${GREEN}Pyarrow installation complete${NC}"

echo -e "\n${YELLOW}Installing fastparquet...${NC}"
conda install fastparquet -y -c conda-forge
echo -e "\n${GREEN}Fastparquet installation complete${NC}"

echo -e "\n${YELLOW}Installing pecanpy...${NC}"
# pecanpy is not available via conda so install using pip
pip install pecanpy
echo -e "\n${GREEN}Pecanpy installation complete${NC}"

echo -e "\n${GREEN}All dependencies have been successfully installed in the conda environment '${ENV_NAME}'${NC}"
echo -e "\nTo activate the environment later, run: conda activate ${ENV_NAME}"

####################################################################################################
