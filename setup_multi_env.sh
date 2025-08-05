#!/bin/bash
# Setup script for multi-environment Automated MP Pipeline

echo "Setting up Multi-Environment Automated MP Pipeline"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

# Check if we're in the right directory
if [ ! -f "main_pipeline.py" ]; then
    echo "ERROR: Please run this script from the automated-mp-pipeline directory"
    exit 1
fi

# Function to create and setup an environment
setup_environment() {
    local env_name=$1
    local requirements_file=$2
    local description=$3
    
    echo ""
    echo "Setting up $description environment ($env_name)..."
    echo "---------------------------------------------------"
    
    # Remove existing environment if it exists
    if [ -d "$env_name" ]; then
        echo "Removing existing $env_name environment..."
        rm -rf "$env_name"
    fi
    
    # Create virtual environment
    echo "Creating $env_name virtual environment..."
    python3 -m venv "$env_name"
    
    # Activate and setup
    echo "Activating $env_name and installing dependencies..."
    source "$env_name/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "$requirements_file" ]; then
        echo "Installing from $requirements_file..."
        pip install -r "$requirements_file"
    else
        echo "WARNING: $requirements_file not found, skipping dependency installation"
    fi
    
    # Deactivate
    deactivate
    
    echo "✓ $description environment setup complete"
}

# Setup individual environments
setup_environment "env_cv" "requirements_cv.txt" "Computer Vision"
setup_environment "env_yochlol" "requirements_yochlol.txt" "Time Series (yochlol)"
setup_environment "env_tabular" "tabular_modeling/requirements.txt" "Tabular Modeling"

# Create main pipeline environment (minimal, just for orchestration)
echo ""
echo "Setting up main pipeline orchestration environment..."
echo "---------------------------------------------------"

if [ -d "env_main" ]; then
    echo "Removing existing env_main environment..."
    rm -rf env_main
fi

python3 -m venv env_main
source env_main/bin/activate

pip install --upgrade pip
pip install pyyaml click pathlib2 python-dateutil tqdm pytest

deactivate

echo "✓ Main pipeline environment setup complete"

# Test all environments
echo ""
echo "Testing environment imports..."
echo "============================="

test_environment() {
    local env_name=$1
    local description=$2
    local test_imports=$3
    
    echo "Testing $description ($env_name)..."
    source "$env_name/bin/activate"
    
    python3 -c "$test_imports" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ $description environment working"
    else
        echo "✗ $description environment has issues"
    fi
    
    deactivate
}

# Test imports for each environment
test_environment "env_cv" "Computer Vision" "import numpy, pandas, yaml, cv2, boto3; print('CV environment OK')"
test_environment "env_yochlol" "Time Series" "import numpy, pandas, torch, flaml, matplotlib; print('Yochlol environment OK')"
test_environment "env_tabular" "Tabular Modeling" "import numpy, pandas, sklearn, xgboost, boto3; print('Tabular environment OK')"
test_environment "env_main" "Main Pipeline" "import yaml, click; print('Main environment OK')"

echo ""
echo "=================================================="
echo "Multi-environment setup complete!"
echo ""
echo "Available environments:"
echo "  env_cv/       - Computer vision processing"
echo "  env_yochlol/  - Time series analysis (yochlol)"
echo "  env_tabular/  - Tabular modeling"
echo "  env_main/     - Main pipeline orchestration"
echo ""
echo "To activate an environment:"
echo "  source env_yochlol/bin/activate"
echo ""
echo "To run the full pipeline:"
echo "  python multi_env_pipeline.py"
echo "=================================================="