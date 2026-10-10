# Finds GpFlash's dependencies, for the root CMakeLists.txt and for the installed package config, so the two cannot
# disagree. The includer defines gpflash_find_dependency (find_package REQUIRED in the build, find_dependency in the
# config) and sets GPFLASH_BLA_VENDOR. This file sets BLA_VENDOR and BLA_SIZEOF_INTEGER from it; the includer restores
# the caller's values.

# Apple clang has no OpenMP of its own: point FindOpenMP at one, e.g. -DOpenMP_ROOT=$(brew --prefix libomp) (README).
# REQUIRED: the OpenMP pragmas in the headers are unguarded.
gpflash_find_dependency(OpenMP)

# Through CMake's FindBLAS and FindLAPACK, because OpenBLAS's own package config names paths on the machine that built
# it. Both read BLA_VENDOR. BLAS is found and linked itself, not left to LAPACK's link line: GpFlash calls BLAS
# routines directly, and a LAPACK built apart from its BLAS, as on Windows, does not carry it.
# Only a BLAS taking 32-bit integers, which arma::blas_int is and BlasLapack.hpp's prototypes pass. A 64-bit-integer
# build (ILP64: openblas64, MKL's ilp64 libraries) would read every dimension wrong without failing anywhere, so
# FindBLAS is told the width and refuses one.
set(BLA_VENDOR "${GPFLASH_BLA_VENDOR}")
set(BLA_SIZEOF_INTEGER 4)
gpflash_find_dependency(BLAS)
gpflash_find_dependency(LAPACK)

# The cache format's storage, through the C API.
gpflash_find_dependency(HDF5 COMPONENTS C)

# The optimizer backend behind OptimizerNlopt.hpp, which uses nlopt.hpp only. A distribution may install NLopt's
# C++-built variant, whose config names NLopt::nlopt_cxx, so either target serves.
gpflash_find_dependency(NLopt CONFIG)
if(NOT TARGET GpFlash::NLopt)
    if(TARGET NLopt::nlopt)
        set(_gpflash_nlopt_target NLopt::nlopt)
    elseif(TARGET NLopt::nlopt_cxx)
        set(_gpflash_nlopt_target NLopt::nlopt_cxx)
    else()
        message(FATAL_ERROR "GpFlash: NLopt's package config defines neither NLopt::nlopt nor NLopt::nlopt_cxx.")
    endif()
    add_library(GpFlash::NLopt INTERFACE IMPORTED)
    set_target_properties(GpFlash::NLopt PROPERTIES INTERFACE_LINK_LIBRARIES ${_gpflash_nlopt_target})
    unset(_gpflash_nlopt_target)
endif()

# Headers only: ARMA_DONT_USE_WRAPPER binds every BLAS/LAPACK call straight to the linked library, so libarmadillo is
# not linked. Through CMake's FindArmadillo, because Armadillo's own package config names paths on the machine that
# built it. BlasLapack.hpp enforces the same version floor for a consumer not using CMake; change both together.
gpflash_find_dependency(Armadillo 9.870)
if(NOT TARGET GpFlash::Armadillo)
    add_library(GpFlash::Armadillo INTERFACE IMPORTED)
    set_target_properties(GpFlash::Armadillo PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ARMADILLO_INCLUDE_DIRS}")
endif()
