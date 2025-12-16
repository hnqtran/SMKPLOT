#!/bin/bash

# SMKPLOT Setup Script
# This script sets up a local Python virtual environment and configures the tool to use it.

# Configuration
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
MAIN_SCRIPT="main.py"

# Ensure we are in the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "   SMKPLOT Environment Setup"
echo "========================================================"

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH."
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating virtual environment ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
else
    echo "[1/3] Virtual environment ($VENV_DIR) already exists."
fi

# 3. Install Dependencies
echo "[2/3] Installing dependencies from $REQUIREMENTS..."
PIP_EXEC="$SCRIPT_DIR/$VENV_DIR/bin/pip"
PYTHON_EXEC="$SCRIPT_DIR/$VENV_DIR/bin/python"

"$PIP_EXEC" install --upgrade pip > /dev/null
if [ -f "$REQUIREMENTS" ]; then
    "$PIP_EXEC" install -r "$REQUIREMENTS"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install requirements."
        exit 1
    fi
else
    echo "WARNING: $REQUIREMENTS not found. Skipping package installation."
fi

# 4. Update Shebang in main.py
echo "[3/3] Configuring $MAIN_SCRIPT..."
if [ -f "$MAIN_SCRIPT" ]; then
    # Create a temporary file with the new shebang
    echo "#!$PYTHON_EXEC" > "${MAIN_SCRIPT}.tmp"
    
    # Append the original file content, skipping the first line (old shebang)
    # We use 'sed' to print from line 2 to end to avoid issues if file is < 2 lines
    sed -n '2,$p' "$MAIN_SCRIPT" >> "${MAIN_SCRIPT}.tmp"
    
    # Replace the original file
    mv "${MAIN_SCRIPT}.tmp" "$MAIN_SCRIPT"
    
    # Make executable
    chmod +x "$MAIN_SCRIPT"
    echo "      Updated shebang to: $PYTHON_EXEC"
else
    echo "ERROR: $MAIN_SCRIPT not found."
    exit 1
fi

echo "========================================================"
echo "Setup Complete!"
echo "You can now run the tool using: ./$MAIN_SCRIPT"
echo "========================================================"
