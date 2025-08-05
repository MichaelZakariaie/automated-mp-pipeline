#!/bin/bash
# Setup script for the Automated MP Pipeline test environment (minimal version)

echo "Setting up Automated MP Pipeline Test Environment (Minimal)"
echo "=========================================================="
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

# Remove old test environment if it exists
if [ -d "test_env" ]; then
    echo "Removing old test environment..."
    rm -rf test_env
fi

# Create virtual environment
echo ""
echo "Creating test virtual environment..."
python3 -m venv test_env
echo "Test virtual environment created."

# Activate virtual environment
echo ""
echo "Activating test virtual environment..."
source test_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install minimal dependencies
echo ""
echo "Installing minimal dependencies for testing..."
echo "This may take a few minutes..."

# Install PyTorch CPU version first (to avoid CUDA dependencies)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing remaining dependencies..."
pip install -r requirements_minimal.txt

# Verify critical imports work
echo ""
echo "Verifying critical imports..."

python3 << 'EOF'
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
        print('✓ {}'.format(module))
    except ImportError as e:
        print('✗ {}: {}'.format(module, e))
        failed_imports.append(module)

if failed_imports:
    print('\nWARNING: Failed to import: {}'.format(', '.join(failed_imports)))
    sys.exit(1)
else:
    print('\nAll critical imports successful!')
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "Test environment setup complete!"
    echo ""
    echo "To run tests:"
    echo "1. source test_env/bin/activate"
    echo "2. python test_pipeline.py"
    echo ""
    echo "To deactivate virtual environment later:"
    echo "deactivate"
    echo "=========================================================="
else
    echo ""
    echo "ERROR: Some dependencies failed to install properly."
    echo "Please check the error messages above and try again."
    exit 1
fi