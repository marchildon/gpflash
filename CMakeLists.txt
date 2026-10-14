# The package's one build: standalone, from a parent project's add_subdirectory, or by a build system that puts the
# dependencies on CMAKE_PREFIX_PATH.

cmake_minimum_required(VERSION 3.22)

# The version is set once, in include/gpflash/Version.hpp; CMake and pyproject.toml both read it from there.
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/gpflash/Version.hpp _gpflash_version_hpp)
string(REGEX MATCH "MAJOR ([0-9]+)\n#define GPFLASH_VERSION_MINOR ([0-9]+)\n#define GPFLASH_VERSION_PATCH ([0-9]+)"
       _ "${_gpflash_version_hpp}")
project(GpFlash VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# The package's own targets, the tests, compile as ISO C++14 (-std=c++14, not the compiler's gnu++14 default), so a
# GNU extension in a header fails here. Not exported: a consumer only inherits the cxx_std_14 floor below.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GPFLASH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Only 64-bit targets have been built and tested. A 32-bit one should compile, ARMA_64BIT_WORD below keeping the
# element counts 64-bit, and run out of address space at roughly 10,000 training points.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "GpFlash is tested on 64-bit targets only; this one has ${CMAKE_SIZEOF_VOID_P}-byte pointers.")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# The vendor as FindBLAS spells it (OpenBLAS, Apple for Accelerate, Intel10_64lp for MKL, ...). The caller's
# BLA_VENDOR wins; otherwise Accelerate on Apple, which every Mac has, and OpenBLAS elsewhere.
if(DEFINED BLA_VENDOR)
    set(GPFLASH_BLA_VENDOR "${BLA_VENDOR}")
elseif(DEFINED ENV{BLA_VENDOR})
    set(GPFLASH_BLA_VENDOR "$ENV{BLA_VENDOR}")
elseif(APPLE)
    set(GPFLASH_BLA_VENDOR Apple)
else()
    set(GPFLASH_BLA_VENDOR OpenBLAS)
endif()

# ARMA_DONT_USE_WRAPPER binds BlasLapack.hpp's extern "C" prototypes straight to the linked library.
# ARMA_USE_BLAS and ARMA_USE_LAPACK let Armadillo's own operations call the same library.
# ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS stays off: BlasLapack.hpp follows Armadillo's hidden-length setting, and a
# gfortran-built LAPACK needs the lengths (GCC bug 90329).
# ARMA_NO_DEBUG drops Armadillo's bounds checks; the library's own checks guard what reaches its loops.
# ARMA_USE_HDF5 selects the save path posterior_cache/ needs; a consumer compiling without it would get a different
# one that does not work.
# ARMA_64BIT_WORD is Armadillo's default on a 64-bit target, stated because an n_train by n_train buffer passes 2^31
# elements at n_train ~ 46,000. BlasLapack.hpp static_asserts it for a consumer compiling without these defines.
# The target's INTERFACE definitions below hand every one to consumers.
set(GPFLASH_ARMA_DEFS
    ARMA_DONT_USE_WRAPPER
    ARMA_NO_DEBUG
    ARMA_USE_BLAS
    ARMA_USE_LAPACK
    ARMA_USE_HDF5
    ARMA_64BIT_WORD)

# Platform.hpp's probe, decided here where the vendor is known; "All" leaves it to the header. Apple and NAS are
# FindBLAS's names for Accelerate.
if(GPFLASH_BLA_VENDOR MATCHES "^(Apple|NAS)$")
    set(GPFLASH_PLATFORM_DEFS GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW=1)
elseif(GPFLASH_BLA_VENDOR STREQUAL "All")
    set(GPFLASH_PLATFORM_DEFS "")
else()
    set(GPFLASH_PLATFORM_DEFS GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW=0)
endif()

macro(gpflash_find_dependency)
    find_package(${ARGV} REQUIRED)
endmacro()
include(cmake/GpFlashDependencies.cmake)

# Header-only, so an INTERFACE target: a consumer compiles every loop itself, with its own flags. The README's
# "Compiling the headers" section is the contract this target meets; a build not using it follows that section.
add_library(GpFlash INTERFACE)
# The installed name, so a parent project's add_subdirectory links the same target a find_package user does.
add_library(GpFlash::GpFlash ALIAS GpFlash)
target_include_directories(GpFlash INTERFACE
    $<BUILD_INTERFACE:${GPFLASH_ROOT}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(GpFlash INTERFACE ${GPFLASH_ARMA_DEFS} ${GPFLASH_PLATFORM_DEFS})

# A floor, not a pin.
target_compile_features(GpFlash INTERFACE cxx_std_14)

target_link_libraries(GpFlash INTERFACE
    GpFlash::Armadillo
    OpenMP::OpenMP_CXX
    LAPACK::LAPACK
    BLAS::BLAS
    GpFlash::HDF5
    GpFlash::NLopt)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
install(FILES LICENSE NOTICE THIRD-PARTY-NOTICES DESTINATION ${CMAKE_INSTALL_DOCDIR})

# A consumer calls find_package(GpFlash) and links GpFlash::GpFlash. The package config finds the dependencies,
# then includes the targets file.
set(GPFLASH_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/GpFlash)
install(TARGETS GpFlash EXPORT GpFlash-export)
install(EXPORT GpFlash-export
        NAMESPACE GpFlash::
        DESTINATION ${GPFLASH_INSTALL_CMAKEDIR}
        FILE GpFlashTargets.cmake)
configure_file(cmake/GpFlashConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/GpFlashConfig.cmake @ONLY)
# Before 1.0 a minor release may break the interface, so a request for 0.Y accepts only 0.Y.x.
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/GpFlashConfigVersion.cmake
                                 COMPATIBILITY SameMinorVersion
                                 ARCH_INDEPENDENT)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GpFlashConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GpFlashConfigVersion.cmake
        cmake/GpFlashDependencies.cmake
        DESTINATION ${GPFLASH_INSTALL_CMAKEDIR})

# The Python module gpflash._core: built when pip drives this file through scikit-build-core (pyproject.toml), which
# defines SKBUILD, or on request. A C++ consumer never needs Python.
option(GPFLASH_BUILD_PYTHON "Build the Python module gpflash._core" OFF)
if(GPFLASH_BUILD_PYTHON OR DEFINED SKBUILD)
    add_subdirectory(python)
endif()

# Only for this project's own build: a parent project's add_subdirectory builds none of the tests.
if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tst)
    endif()
endif()
