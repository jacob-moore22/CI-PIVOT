# Guard against sourcing
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed
    : # continue with script
else
    echo "This script should be executed, not sourced"
    echo "Please run: ./build.sh -t <build_type>"
    return 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [-t build_type] [-d]"
    echo "build_type options: serial, openmp, pthreads, cuda, hip"
    echo "  -t    Specify build type (required)"
    echo "  -d    Enable debug build (optional)"
    exit 1
}

# Parse command line arguments
while getopts "t:d" opt; do
    case ${opt} in
        t )
            build_type=$OPTARG
            ;;
        d )
            debug=true
            ;;
        \? )
            usage
            ;;
    esac
done

# Validate build type
if [ -z "$build_type" ]; then
    echo "Error: Build type (-t) is required"
    usage
fi


# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build_${build_type}"

# Create the build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"

# Get the directory where the script is called from
CURRENT_DIR=$(pwd)
MATAR_SOURCE_DIR="${CURRENT_DIR}/MATAR"
MATAR_BUILD_DIR="${CURRENT_DIR}/build_tmp/matar"
MATAR_INSTALL_DIR="${CURRENT_DIR}/MATAR/install/matar"
MATAR_BUILD_CORES=$(nproc)

# Clone MATAR repository if it doesn't exist
if [ ! -d "${MATAR_SOURCE_DIR}" ]; then
    echo "Cloning MATAR repository..."
    git clone https://github.com/lanl/MATAR.git
fi

# Verify the repository was cloned successfully
if [ ! -f "${MATAR_SOURCE_DIR}/CMakeLists.txt" ]; then
    echo "Error: Failed to clone MATAR repository or repository is invalid"
    exit 1
fi

# Verify the build script exists
if [ ! -f "${MATAR_SOURCE_DIR}/scripts/build-matar.sh" ]; then
    echo "Error: build-matar.sh not found at ${MATAR_SOURCE_DIR}/scripts/build-matar.sh"
    exit 1
fi

# Source the build script with proper arguments
echo "Building Kokkos..."
source "${MATAR_SOURCE_DIR}/scripts/build-matar.sh" --kokkos_build_type=${build_type} --build_action=install-kokkos

echo "Building MATAR..."
source "${MATAR_SOURCE_DIR}/scripts/build-matar.sh" --kokkos_build_type=${build_type} --build_action=install-matar

echo "Matar installation complete"