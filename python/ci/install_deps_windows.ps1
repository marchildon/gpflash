# Installs the libraries a GpFlash wheel links on Windows, for cibuildwheel's before-all: OpenBLAS, HDF5, Armadillo
# and NLopt, from the sources install_deps_unix.sh pins, with the same SHA-256 checks.
#
# Built with the Visual Studio toolchain, so the wheel carries no Fortran or MinGW runtime: OpenBLAS with clang-cl,
# using its own C translation of LAPACK; HDF5 without compression filters, as the cache stores plain arrays; NLopt
# static and with NLOPT_LUKSAN=OFF, so none of its LGPL code. delvewheel then bundles openblas.dll, hdf5.dll and the
# Microsoft runtimes, which python/LICENSES_bundled.txt lists.
#
# Usage: pwsh -File install_deps_windows.ps1. Everything goes to $env:GPFLASH_DEPS_PREFIX, C:\gpflash_deps by default.
$ErrorActionPreference = "Stop"

$NloptVersion = "2.11.0"
$NloptSha256 = "53e552d83e9294d67db37f0f4a23f15933a9ef698485301a18b98b40004cf0de"
$OpenBlasVersion = "0.3.34"
$OpenBlasSha256 = "cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed"
$Hdf5Version = "1.14.6"
$Hdf5Sha256 = "e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b"
$ArmadilloVersion = "15.6.1"
$ArmadilloSha256 = "23fe3b3848e2929ab39089ad6e8e445d3a24cc39d36c018a253b97e56ba88c2c"

$Prefix = if ($env:GPFLASH_DEPS_PREFIX) { $env:GPFLASH_DEPS_PREFIX } else { "C:\gpflash_deps" }
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) "gpflash_deps_build"

# Written last, so it marks a complete install. The wheel workflow caches the prefix, as OpenBLAS alone takes almost
# two hours to build; a restored, complete prefix is used as it is.
$CompleteMarker = Join-Path $Prefix "gpflash_deps_complete"
if (Test-Path $CompleteMarker) {
    Write-Host "install_deps_windows.ps1: using the dependencies already in $Prefix"
    exit 0
}
New-Item -ItemType Directory -Force -Path $Prefix, $WorkDir | Out-Null

# The Visual Studio developer environment, so cmake with Ninja finds cl, clang-cl and the Windows SDK.
$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsPath) { throw "install_deps_windows.ps1: no Visual Studio with the C++ tools found" }
Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"

# Download, check and extract one source archive; returns the extracted directory. curl rather than
# Invoke-WebRequest, which SourceForge answers with an HTML page instead of the file.
function Get-Source($Name, $Url, $Sha256, $ExtractedDir) {
    $Archive = Join-Path $WorkDir "$Name.archive"
    curl.exe -fsSL --retry 3 -o $Archive $Url
    if ($LASTEXITCODE -ne 0) { throw "$Name download failed" }
    $ActualSha256 = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLower()
    if ($ActualSha256 -ne $Sha256) { throw "$Name archive SHA-256 mismatch: $ActualSha256" }
    tar -xf $Archive -C $WorkDir  # either tar detects gzip or xz itself
    if ($LASTEXITCODE -ne 0) { throw "$Name extraction failed" }
    return Join-Path $WorkDir $ExtractedDir
}

# Configure, build and install with Ninja; Release, into the prefix.
function Build-Source($Name, $SourceDir, $CmakeArgs) {
    $BuildDir = Join-Path $WorkDir "$Name-build"
    cmake -S $SourceDir -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$Prefix" @CmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name configure failed" }
    cmake --build $BuildDir --parallel
    if ($LASTEXITCODE -ne 0) { throw "$Name build failed" }
    cmake --install $BuildDir
    if ($LASTEXITCODE -ne 0) { throw "$Name install failed" }
}

# OpenBLAS: DYNAMIC_ARCH picks the kernels for the CPU at run time, TARGET being the baseline. clang-cl compiles
# the optimized kernels, which cl cannot. C_LAPACK is OpenBLAS's C translation of LAPACK, so no Fortran. Its own
# threads rather than OpenMP, which with clang-cl would mean a second OpenMP runtime beside the module's vcomp140.
$OpenBlasDir = Get-Source "openblas" "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$OpenBlasVersion/OpenBLAS-$OpenBlasVersion.tar.gz" $OpenBlasSha256 "OpenBLAS-$OpenBlasVersion"
Build-Source "openblas" $OpenBlasDir @(
    "-DCMAKE_C_COMPILER=clang-cl",
    "-DCMAKE_MT=mt",
    "-DNOFORTRAN=ON",
    "-DC_LAPACK=ON",
    "-DBUILD_SHARED_LIBS=ON",
    "-DDYNAMIC_ARCH=ON",
    "-DTARGET=PRESCOTT",
    "-DUSE_THREAD=ON",
    "-DUSE_OPENMP=OFF",
    "-DNUM_THREADS=64",
    "-DINTERFACE64=0",
    "-DBUILD_TESTING=OFF",
    "-DBUILD_BENCHMARKS=OFF")

# HDF5: the C library only, without compression filters.
$Hdf5Dir = Get-Source "hdf5" "https://github.com/HDFGroup/hdf5/releases/download/hdf5_$Hdf5Version/hdf5-$Hdf5Version.tar.gz" $Hdf5Sha256 "hdf5-$Hdf5Version"
Build-Source "hdf5" $Hdf5Dir @(
    "-DBUILD_SHARED_LIBS=ON",
    "-DBUILD_STATIC_LIBS=OFF",
    "-DBUILD_TESTING=OFF",
    "-DHDF5_BUILD_TOOLS=OFF",
    "-DHDF5_BUILD_UTILS=OFF",
    "-DHDF5_BUILD_EXAMPLES=OFF",
    "-DHDF5_BUILD_FORTRAN=OFF",
    "-DHDF5_BUILD_CPP_LIB=OFF",
    "-DHDF5_BUILD_HL_LIB=OFF",
    "-DHDF5_BUILD_JAVA=OFF",
    "-DHDF5_ENABLE_Z_LIB_SUPPORT=OFF",
    "-DHDF5_ENABLE_SZIP_SUPPORT=OFF")

# Armadillo: GpFlash uses the headers only (ARMA_DONT_USE_WRAPPER), so they are copied; Armadillo does not support
# its CMake install on Windows. Its unconfigured config.hpp leaves the wrapper off, so FindArmadillo needs no library.
$ArmadilloDir = Get-Source "armadillo" "https://sourceforge.net/projects/arma/files/armadillo-$ArmadilloVersion.tar.xz/download" $ArmadilloSha256 "armadillo-$ArmadilloVersion"
New-Item -ItemType Directory -Force -Path (Join-Path $Prefix "include") | Out-Null
Copy-Item -Recurse -Force (Join-Path $ArmadilloDir "include\*") (Join-Path $Prefix "include")

$NloptDir = Get-Source "nlopt" "https://github.com/stevengj/nlopt/archive/refs/tags/v$NloptVersion.tar.gz" $NloptSha256 "nlopt-$NloptVersion"
Build-Source "nlopt" $NloptDir @(
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DNLOPT_LUKSAN=OFF",
    "-DNLOPT_CXX=OFF",
    "-DNLOPT_FORTRAN=OFF",
    "-DNLOPT_PYTHON=OFF",
    "-DNLOPT_OCTAVE=OFF",
    "-DNLOPT_MATLAB=OFF",
    "-DNLOPT_GUILE=OFF",
    "-DNLOPT_JAVA=OFF",
    "-DNLOPT_SWIG=OFF",
    "-DNLOPT_TESTS=OFF")

Remove-Item -Recurse -Force $WorkDir
Set-Content -Path $CompleteMarker -Value "OpenBLAS $OpenBlasVersion, HDF5 $Hdf5Version, Armadillo $ArmadilloVersion, NLopt $NloptVersion"
Write-Host "install_deps_windows.ps1: dependencies installed in $Prefix"
Get-ChildItem (Join-Path $Prefix "bin") -Filter *.dll | ForEach-Object { Write-Host "  $($_.Name)" }
