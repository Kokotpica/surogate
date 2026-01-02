#!/bin/bash
# Generate Python stub file for the _surogate C++ extension module
# Run this after rebuilding the extension if the C++ API changes

set -e

echo "Generating stub file for surogate._surogate..."
pybind11-stubgen surogate._surogate -o stubs --ignore-all-errors

echo "Moving stub file to package directory..."
cp stubs/surogate/_surogate.pyi surogate/_surogate.pyi

echo "Fixing numpy type hints..."
sed -i "s/ndarray\[dtype=int32, \.\.\., order='C', device='cpu'\]/npt.NDArray[np.int32]/g" surogate/_surogate.pyi

echo "Adding numpy imports..."
sed -i '4a import numpy as np\nimport numpy.typing as npt\n' surogate/_surogate.pyi

echo "Fixing SystemInfo static methods..."
# Note: Manual fixes may be needed for complex types that pybind11-stubgen can't parse
# Check the generated file and update as needed

echo "Cleaning up temporary files..."
rm -rf stubs/

echo "✓ Stub file generated at surogate/_surogate.pyi"
