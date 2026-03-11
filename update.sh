#!/bin/bash
# Author: tranhuy@email.unc.edu

# SMKPLOT Update Script
# This script forcefully pulls the latest updates from the GitHub repository and reinstalls the environment.
# WARNING: This will discard any local modifications made to tracking files.

# Ensure we are in the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "   SMKPLOT Update Script"
echo "========================================================"

echo "WARNING: This will discard any local changes to tracked files."
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Update aborted by user."
    exit 1
fi

echo "[1/3] Discarding local modifications..."
git reset --hard origin/main
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to reset local branch. Are you connected to the repository?"
    exit 1
fi

echo "[2/3] Pulling latest code from origin/main..."
git pull origin main
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to pull updates."
    exit 1
fi

echo "[3/3] Re-running installation to update hooks and shebangs..."
if [ -x "./install.sh" ]; then
    ./install.sh
else
    echo "ERROR: install.sh not found or not executable. Please run it manually."
    exit 1
fi

echo "========================================================"
echo "Update Complete!"
echo "========================================================"
