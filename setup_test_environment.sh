#!/bin/bash
# Setup script for the Automated MP Pipeline test environment

echo "Setting up Automated MP Pipeline Test Environment"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

# Check if we're in the right directory
if [ ! -f "test_pipeline.py" ]; then
    echo "ERROR: Please run this script from the automated-mp-pipeline directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "test_env" ]; then
    echo ""
    echo "Creating test virtual environment..."
    python3 -m venv test_env
    echo "Test virtual environment created."
else
    echo ""
    echo "Test virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating test virtual environment..."
source test_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from comprehensive requirements
echo ""
echo "Installing comprehensive dependencies..."
echo "This may take several minutes..."

# Install in stages to handle potential conflicts
echo "Installing core dependencies..."
pip install numpy pandas pyyaml tqdm joblib

echo "Installing AWS dependencies..."
pip install boto3 awswrangler botocore

echo "Installing ML dependencies..."
pip install scikit-learn scipy xgboost

echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing visualization dependencies..."
pip install matplotlib seaborn plotly

echo "Installing data processing dependencies..."
pip install pyarrow

echo "Installing testing dependencies..."
pip install pytest pytest-cov

echo "Installing remaining dependencies..."
pip install -r requirements_comprehensive.txt

# Verify critical imports work
echo ""
echo "Verifying critical imports..."

python3 -c "
import sys
import traceback

modules_to_test = [
    'numpy', 'pandas', 'yaml', 'boto3', 'sklearn', 
    'torch', 'matplotlib', 'pytest'
]

failed_imports = []

for module in modules_to_test:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
        failed_imports.append(module)

if failed_imports:
    print(f'\\nWARNING: Failed to import: {', '.join(failed_imports)}')
    sys.exit(1)
else:
    print('\\nAll critical imports successful!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Test environment setup complete!"
    echo ""
    echo "To run tests:"
    echo "1. source test_env/bin/activate"
    echo "2. python test_pipeline.py"
    echo ""
    echo "To deactivate virtual environment later:"
    echo "deactivate"
    echo "================================================"
else
    echo ""
    echo "ERROR: Some dependencies failed to install properly."
    echo "Please check the error messages above and try again."
    exit 1
fi