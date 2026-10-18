# The Python module gpflash._core: pybind11 bindings over the header-only library, compiled as ISO C++14 like the rest
# of the package. pyproject.toml drives this through scikit-build-core; only its install component, python, goes into
# the wheel, so the headers and CMake package files the root installs stay out of it.

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.0 CONFIG REQUIRED)  # 3.0 for py::native_enum; pyproject.toml requires the same

pybind11_add_module(gpflash_python MODULE src/bindings.cpp)
set_target_properties(gpflash_python PROPERTIES OUTPUT_NAME _core)

# GpFlash::GpFlash carries every define, flag and library the headers need.
target_link_libraries(gpflash_python PRIVATE GpFlash::GpFlash)

# The README's "Compiling the headers" flags: -funroll-loops for batch posterior evaluation, -fno-math-errno so the
# kernels' math functions vectorize. Never -ffast-math, which core/Platform.hpp rejects.
target_compile_options(gpflash_python PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-funroll-loops -fno-math-errno>)

install(TARGETS gpflash_python LIBRARY DESTINATION gpflash COMPONENT python)
