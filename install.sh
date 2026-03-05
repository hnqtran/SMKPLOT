#!/bin/bash
# Author: tranhuy@email.unc.edu

# SMKPLOT Setup Script
# This script sets up a local Python virtual environment and configures the tool to use it.

# Configuration
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
MAIN_SCRIPT="smkplot.py"

# Ensure we are in the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "   SMKPLOT Environment Setup"
echo "========================================================"

# 1. Check for Python 3.9+
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -ne 3 ] || [ "$MINOR" -lt 8 ]; then
    echo "ERROR: SMKPLOT requires Python 3.8 or higher."
    echo "       Detected version: $PYTHON_VERSION"
    echo "       Please use a newer Python version (e.g. 3.8, 3.9, 3.10, 3.11, 3.12)."
    exit 1
fi
echo "      Confirmed Python $PYTHON_VERSION"

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

"$PIP_EXEC" install --upgrade pip setuptools wheel > /dev/null
if [ -f "$REQUIREMENTS" ]; then
    "$PIP_EXEC" install -r "$REQUIREMENTS"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install requirements."
        exit 1
    fi
else
    echo "WARNING: $REQUIREMENTS not found. Skipping package installation."
fi

# 4. Update Shebang in smkplot.py and utilities
echo "[3/3] Configuring $MAIN_SCRIPT..."
if [ -f "$MAIN_SCRIPT" ]; then
    # Create a temporary file with the new shebang
    echo "#!$PYTHON_EXEC" > "${MAIN_SCRIPT}.tmp"
    
    # Append the original file content, skipping the first line (old shebang)
    sed -n '2,$p' "$MAIN_SCRIPT" >> "${MAIN_SCRIPT}.tmp"
    
    # Replace the original file
    mv "${MAIN_SCRIPT}.tmp" "$MAIN_SCRIPT"
    
    # Make executable
    chmod +x "$MAIN_SCRIPT"
    echo "      Updated shebang to: $PYTHON_EXEC"

    # Also update optional GUI/Utility scripts if they exist
    for script in "gui_qt.py" "ncf_processing.py" "batch.py"; do
        if [ -f "$script" ]; then
             echo "#!$PYTHON_EXEC" > "${script}.tmp"
             sed -n '2,$p' "$script" >> "${script}.tmp"
             mv "${script}.tmp" "$script"
             chmod +x "$script"
             echo "      Updated shebang for $script"
        fi
    done
else
    echo "ERROR: $MAIN_SCRIPT not found."
    exit 1
fi

# 5. Final Capability Check
echo "========================================================"
echo "Checking GUI capabilities..."

# Check Qt (Preferred)
"$PYTHON_EXEC" -c "import PySide6" &> /dev/null
PYSIDE_STATUS=$?
"$PYTHON_EXEC" -c "import PyQt5" &> /dev/null
PYQT_STATUS=$?

if [ $PYSIDE_STATUS -eq 0 ]; then
    echo "SUCCESS: Qt GUI (PySide6) detected. Modern GUI is ready."
elif [ $PYQT_STATUS -eq 0 ]; then
    echo "SUCCESS: Qt GUI (PyQt5) detected. Compatibility GUI is ready."
else
    # Fallback check for Tkinter
    "$PYTHON_EXEC" -c "import tkinter" &> /dev/null
    if [ $? -eq 0 ]; then
        echo "INFO: Tkinter detected. Legacy GUI is available."
    else
        echo "WARNING: No GUI libraries (Qt or Tk) detected."
        echo "         Only Batch Mode (--run-mode batch) will be available."
    fi
fi

# Check for Display
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "NOTE: No X11/Wayland display detected. Use --run-mode batch for headless operation."
fi

echo "========================================================"
echo "Setup Complete!"
echo "Run the tool using: ./$MAIN_SCRIPT"
echo "========================================================"
