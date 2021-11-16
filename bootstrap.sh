#!/usr/bin/env bash
set -euo pipefail

# Directories to use
SRC_DIR="$(cd $(dirname "$0") && pwd)"
ENV_DIR="$(pwd)/venv"
NUMPROC=$(nproc)

# Specific commits to checkout
SBLU_COMMIT=0cdd4ef
PRODY_VERSION=1.10

# Setup conda env
if [ '1' ]; then

if [ ! -d "${ENV_DIR}" ]; then
    conda env create -f "${SRC_DIR}/conda-env.yml" --prefix "${ENV_DIR}"
else
    conda env update -f "${SRC_DIR}/conda-env.yml" --prefix "${ENV_DIR}"
fi

# Create a separate conda env for mgltools
if [ ! -d "${ENV_DIR}/mgltools_env" ]; then
  conda create --prefix "${ENV_DIR}/mgltools_env" -c bioconda mgltools=1.5.7 -y
fi

fi

# Create conda environment in the current directory
set +u  # conda references PS1 variable that is not set in scripts
source activate "${ENV_DIR}"
set -u

# setting env variables
set +u
export PATH="${ENV_DIR}/bin:${ENV_DIR}/lib:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${ENV_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}"
set -u

# Install ProDy
pip install prody==${PRODY_VERSION}

# Install sb-lab-utils
rm -rf sb-lab-utils
git clone https://bitbucket.org/bu-structure/sb-lab-utils.git
cd sb-lab-utils
git checkout $SBLU_COMMIT
pip install -r requirements/pipeline.txt
python setup.py install
cd ../
rm -rf sb-lab-utils

# Install psfgen
cp "${SRC_DIR}/deps/psfgen_1.6.5_Linux-x86_64-multicore" "${ENV_DIR}/bin/psfgen"
chmod u+x "${ENV_DIR}/bin/psfgen"

pip install -e ${SRC_DIR}

echo "Done. To use the new environment, call \`source \"${ENV_DIR}/bin/activate\"\`"
