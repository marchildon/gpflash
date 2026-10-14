#!/usr/bin/env bash
# Installs the libraries a GpFlash wheel links, for cibuildwheel's before-all on Linux (manylinux) and macOS.
#
# Every source is pinned by version and SHA-256. NLopt is built from source on both, static and with
# NLOPT_LUKSAN=OFF, so the wheel carries none of its LGPL code; GpFlash uses only SLSQP (BSD) and CCSA (MIT).
#
#   Linux: OpenBLAS (LAPACK in C, so no libgfortran), HDF5 and Armadillo, built into the prefix.
#   macOS: Accelerate for BLAS/LAPACK; HDF5 and Armadillo built into the prefix, as on Linux; Homebrew's libomp, the
#          one library taken as installed. delocate bundles HDF5 and libomp.
#
# Usage: GPFLASH_DEPS_PREFIX=<dir> bash install_deps_unix.sh. The prefix goes on CMAKE_PREFIX_PATH for the wheel build.
set -euo pipefail

NLOPT_VERSION=2.11.0
NLOPT_SHA256=53e552d83e9294d67db37f0f4a23f15933a9ef698485301a18b98b40004cf0de
OPENBLAS_VERSION=0.3.34
OPENBLAS_SHA256=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed
HDF5_VERSION=1.14.6
HDF5_SHA256=e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b
ARMADILLO_VERSION=15.6.1
ARMADILLO_SHA256=23fe3b3848e2929ab39089ad6e8e445d3a24cc39d36c018a253b97e56ba88c2c

case "$(uname -s)" in
    Linux) default_prefix=/opt/gpflash_deps ;;
    Darwin) default_prefix=/tmp/gpflash_deps ;;
    *) echo "install_deps_unix.sh: unsupported system $(uname -s)" >&2; exit 1 ;;
esac
PREFIX="${GPFLASH_DEPS_PREFIX:-$default_prefix}"
WORK_DIR="$(mktemp -d)"
N_JOB="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
mkdir -p "$PREFIX"

# Download url to file and check its SHA-256.
fetch() {
    local url="$1" sha256="$2" file="$3"
    curl -fsSL --retry 3 -o "$WORK_DIR/$file" "$url"
    if command -v sha256sum >/dev/null; then
        echo "$sha256  $WORK_DIR/$file" | sha256sum -c -
    else
        echo "$sha256  $WORK_DIR/$file" | shasum -a 256 -c -
    fi
}

build_nlopt() {
    fetch "https://github.com/stevengj/nlopt/archive/refs/tags/v${NLOPT_VERSION}.tar.gz" "$NLOPT_SHA256" nlopt.tar.gz
    tar -xzf "$WORK_DIR/nlopt.tar.gz" -C "$WORK_DIR"
    cmake -S "$WORK_DIR/nlopt-${NLOPT_VERSION}" -B "$WORK_DIR/nlopt-build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DNLOPT_LUKSAN=OFF \
        -DNLOPT_CXX=OFF \
        -DNLOPT_FORTRAN=OFF \
        -DNLOPT_PYTHON=OFF \
        -DNLOPT_OCTAVE=OFF \
        -DNLOPT_MATLAB=OFF \
        -DNLOPT_GUILE=OFF \
        -DNLOPT_JAVA=OFF \
        -DNLOPT_SWIG=OFF \
        -DNLOPT_TESTS=OFF
    cmake --build "$WORK_DIR/nlopt-build" --parallel "$N_JOB"
    cmake --install "$WORK_DIR/nlopt-build"
}

build_openblas() {
    fetch "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
        "$OPENBLAS_SHA256" openblas.tar.gz
    tar -xzf "$WORK_DIR/openblas.tar.gz" -C "$WORK_DIR"
    # DYNAMIC_ARCH picks the kernels for the CPU at run time; TARGET is only the baseline. NOFORTRAN builds LAPACK
    # from its C translation, so the wheel needs no libgfortran. USE_OPENMP shares GpFlash's OpenMP runtime rather
    # than starting a second thread pool. 32-bit integers (INTERFACE64=0), as GpFlash requires.
    local target
    case "$(uname -m)" in
        x86_64) target=PRESCOTT ;;
        aarch64) target=ARMV8 ;;
        *) echo "install_deps_unix.sh: no OpenBLAS target for $(uname -m)" >&2; exit 1 ;;
    esac
    make -C "$WORK_DIR/OpenBLAS-${OPENBLAS_VERSION}" -j "$N_JOB" \
        DYNAMIC_ARCH=1 TARGET="$target" USE_OPENMP=1 NOFORTRAN=1 INTERFACE64=0 NUM_THREADS=64 NO_STATIC=1 \
        libs netlib shared
    make -C "$WORK_DIR/OpenBLAS-${OPENBLAS_VERSION}" PREFIX="$PREFIX" NO_STATIC=1 install
}

build_hdf5() {
    fetch "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz" \
        "$HDF5_SHA256" hdf5.tar.gz
    tar -xzf "$WORK_DIR/hdf5.tar.gz" -C "$WORK_DIR"
    # The C library only, without compression filters: the posterior cache stores plain arrays.
    cmake -S "$WORK_DIR/hdf5-${HDF5_VERSION}" -B "$WORK_DIR/hdf5-build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_STATIC_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_UTILS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_BUILD_HL_LIB=OFF \
        -DHDF5_BUILD_JAVA=OFF \
        -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF \
        -DHDF5_ENABLE_SZIP_SUPPORT=OFF
    cmake --build "$WORK_DIR/hdf5-build" --parallel "$N_JOB"
    cmake --install "$WORK_DIR/hdf5-build"
    # HDF5 names its macOS library @rpath/..., which delocate cannot resolve; name it by its full path instead, as
    # Homebrew's libraries are, so delocate finds it to bundle.
    if [ "$(uname -s)" = Darwin ]; then
        local lib
        for lib in "$PREFIX"/lib/libhdf5.*.dylib; do
            [ -L "$lib" ] || install_name_tool -id "$lib" "$lib"
        done
    fi
}

build_armadillo() {
    fetch "https://sourceforge.net/projects/arma/files/armadillo-${ARMADILLO_VERSION}.tar.xz/download" \
        "$ARMADILLO_SHA256" armadillo.tar.xz
    tar -xJf "$WORK_DIR/armadillo.tar.xz" -C "$WORK_DIR"
    # GpFlash uses the headers only (ARMA_DONT_USE_WRAPPER), but CMake's FindArmadillo also requires the small
    # wrapper library, so it is built and installed too; the wheel does not link it.
    cmake -S "$WORK_DIR/armadillo-${ARMADILLO_VERSION}" -B "$WORK_DIR/armadillo-build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DBUILD_SMOKE_TEST=OFF
    cmake --build "$WORK_DIR/armadillo-build" --parallel "$N_JOB"
    cmake --install "$WORK_DIR/armadillo-build"
}

case "$(uname -s)" in
    Linux)
        build_openblas
        build_hdf5
        build_armadillo
        ;;
    Darwin)
        brew install --quiet libomp
        build_hdf5
        build_armadillo
        ;;
esac
build_nlopt

rm -rf "$WORK_DIR"
echo "install_deps_unix.sh: dependencies installed in $PREFIX"
