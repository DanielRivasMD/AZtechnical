#!/bin/bash
####################################################################################################

# Exit immediately if a command exits with a non-zero status
set -e

# Create a virtual environment named 'venv'
python3 -m venv venv

# Upgrade pip within the 'venv'
venv/bin/pip install --upgrade pip

# Install each required package using the pip from the virtual environment
#!/bin/bash
set -e

# Define ANSI color codes
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m"  # No Color

echo -e "\n${YELLOW}Installing pandas...${NC}"
venv/bin/pip install pandas
echo -e "\n${GREEN}Pandas installation complete${NC}"

echo -e "\n${YELLOW}Installing numpy...${NC}"
venv/bin/pip install numpy
echo -e "\n${GREEN}Numpy installation complete${NC}"

echo -e "\n${YELLOW}Installing scikit-learn...${NC}"
venv/bin/pip install scikit-learn
echo -e "\n${GREEN}scikit-learn installation complete${NC}"

echo -e "\n${YELLOW}Installing pecanpy...${NC}"
venv/bin/pip install pecanpy
echo -e "\n${GREEN}pecanpy installation complete${NC}"

echo -e "\n${YELLOW}Installing matplotlib...${NC}"
venv/bin/pip install matplotlib
echo -e "\n${GREEN}matplotlib installation complete${NC}"


echo -e "\n${GREEN}All dependencies have been successfully installed in the 'venv' virtual environment${NC}"

# To start using the virtual environment, run: activat (alias: source venv/bin/activate)


####################################################################################################


