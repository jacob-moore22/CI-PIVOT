#!/bin/bash
# RUN USING: sudo ./install_likwid.sh


echo "Installing LIKWID prerequisites..."
# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

# Install prerequisites (for Debian/Ubuntu systems)
if command -v apt-get >/dev/null; then
    apt-get update
    apt-get install -y build-essential gcc make git
fi

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
# Enable access daemon
sed -i 's/^ACCESSMODE ?= direct/ACCESSMODE ?= accessdaemon/' config.mk

echo "Building LIKWID..."
make

echo "Installing LIKWID..."
make install

# Create necessary directories for performance groups
echo "Setting up performance group directories..."
mkdir -p /usr/local/share/likwid/perfgroups
mkdir -p /usr/local/share/likwid/perfgroups/zen3
mkdir -p /usr/local/share/likwid/perfgroups/zen4
mkdir -p /usr/local/share/likwid/perfgroups/zen5

# Copy performance group files
echo "Copying performance group files..."
cp -r groups/* /usr/local/share/likwid/perfgroups/

# Set proper permissions
chmod -R 755 /usr/local/share/likwid/perfgroups

echo "Enabling machine-specific registers (MSR) for LIKWID..."
modprobe msr

# Create user-specific LIKWID directory
echo "Setting up user-specific LIKWID directory..."
mkdir -p ~/.likwid/groups
cp -r groups/* ~/.likwid/groups/

echo "LIKWID installation completed!"


echo "Installing Valgrind "
sudo snap install valgrind --classic

echo "Installing kcachegrind"
sudo snap install kcachegrind

echo "Installing OpenMPI"
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev