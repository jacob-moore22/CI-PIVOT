#!/bin/bash

# Exit on error
# set -e

# echo "Installing LIKWID prerequisites..."
# # Check if running as root/sudo
# if [ "$EUID" -ne 0 ]; then
#     echo "Please run as root or with sudo"
#     exit 1
# fi

# # Install prerequisites (for Debian/Ubuntu systems)
# if command -v apt-get >/dev/null; then
#     apt-get update
#     apt-get install -y build-essential gcc make git
# fi

# Create a directory for installation
INSTALL_DIR="/tmp/likwid_install"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

echo "Cloning LIKWID repository..."
git clone https://github.com/RRZE-HPC/likwid.git
cd likwid

echo "Configuring LIKWID..."
# Modify config.mk to set installation prefix to /usr/local
sed -i 's/^PREFIX ?= \/usr\/local/PREFIX ?= \/usr\/local/' config.mk
# # Enable access daemon
sed -i 's/^ACCESSMODE ?= direct/ACCESSMODE ?= accessdaemon/' config.mk

echo "Building LIKWID..."
make

echo "Installing LIKWID..."
make install

echo "Cleaning up..."
cd ../..
rm -rf $INSTALL_DIR

echo "LIKWID installation completed!"