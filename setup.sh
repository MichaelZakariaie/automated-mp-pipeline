#!/bin/bash
# Setup script for the Automated MP Pipeline

echo "Automated MP Pipeline Setup"
echo "=========================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

pip install --upgrade pip
pip install -r requirements.txt

# Check if submodules have their own requirements
if [ -f "yochlol/requirements.txt" ]; then
    echo ""
    echo "Installing yochlol dependencies..."
    pip install -r yochlol/requirements.txt
fi

if [ -f "tabular_modeling/requirements.txt" ]; then
    echo ""
    echo "Installing tabular_modeling dependencies..."
    pip install -r tabular_modeling/requirements.txt
fi

# Create necessary directories
echo ""
echo "Creating output directories..."
mkdir -p pipeline_output pipeline_temp pipeline_reports

# Check AWS configuration
echo ""
echo "Checking AWS configuration..."
if aws sts get-caller-identity >/dev/null 2>&1; then
    echo "AWS credentials found and valid."
else
    echo "WARNING: AWS credentials not configured or invalid."
    echo "Run 'aws configure' to set up AWS access if using AWS mode."
fi

# Display next steps
echo ""
echo "=========================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and edit pipeline_config.yaml"
echo "2. Run './run_time_series.sh' to start time series analysis"
echo "3. After external team processes data, run './run_tabular.sh'"
echo ""
echo "For more information, see README.md"