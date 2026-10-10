# The package's tests and checks. Every one runs under ctest.

# Listed explicitly so adding a test is deliberate.
set(GPFLASH_TESTS
    test_gpflash_cache
    test_gpflash_chol_solves
    test_gpflash_inner_product_gradient
    test_gpflash_input_validation
    test_gpflash_kernel_assembly
    test_gpflash_kernel_contract
    test_gpflash_kernel_finite_difference
    test_gpflash_kernel_matrix_grad
    test_gpflash_kernel_center
    test_gpflash_kernel_posterior_assembly
    test_gpflash_kmat_nugget
    test_gpflash_kmat_nugget_grad
    test_gpflash_linear
    test_gpflash_local_optz
    test_gpflash_multistart
    test_gpflash_multistart_report
    test_gpflash_optimizer
    test_gpflash_optimizer_periodic
    test_gpflash_periodic
    test_gpflash_posterior
    test_gpflash_posterior_nonradial
    test_gpflash_recursive_inv
    test_gpflash_search_space
    test_gpflash_sigma_matrix
    test_gpflash_stage_timing
    test_gpflash_string_format
    test_gpflash_struct_checks
    test_gpflash_symmetry
    test_gpflash_training
    test_gpflash_version)

# Its own target, below: GPFLASH_HAVE_MPI changes what the headers compile, and it needs a launcher.
set(GPFLASH_MPI_TEST test_gpflash_optimizer_mpi)

# A source not listed would never run.
file(GLOB GPFLASH_TEST_SOURCES CONFIGURE_DEPENDS
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_gpflash_*.cpp)
foreach(_test_source IN LISTS GPFLASH_TEST_SOURCES)
    get_filename_component(_test_name ${_test_source} NAME_WE)
    if(NOT _test_name IN_LIST GPFLASH_TESTS AND NOT _test_name STREQUAL GPFLASH_MPI_TEST)
        message(FATAL_ERROR "tst/${_test_source} is not in GPFLASH_TESTS, so it would never run.")
    endif()
endforeach()

# Here rather than on the library target, so the package builds warning-clean without imposing these on consumers.
# Not the -funroll-loops a consumer should add: the tests check results, not speed.
# Errors in a Coverage build, the one that gates changes here, and elsewhere only on request, so a newer compiler's
# new warnings do not fail someone else's build.
option(GPFLASH_WARNINGS_AS_ERRORS "Fail the test build on any compiler warning, as a Coverage build always does" OFF)
if(MSVC)
    set(GPFLASH_WARNING_FLAGS /W4)
    set(GPFLASH_WARNING_ERROR_FLAG /WX)
else()
    set(GPFLASH_WARNING_FLAGS -Wall -Wextra)
    set(GPFLASH_WARNING_ERROR_FLAG -Werror)
endif()
if(GPFLASH_WARNINGS_AS_ERRORS OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
    list(APPEND GPFLASH_WARNING_FLAGS ${GPFLASH_WARNING_ERROR_FLAG})
endif()

# One test executable. GpFlash brings include/, its defines, OpenMP and the libraries; this directory is for
# GpFlashTestSupport.hpp. GPFLASH_DEBUG_CHECKS compiles in the GPFLASH_CHECK checks on intermediate calculations,
# so they run on every test run instead of rotting in a separate debug build.
function(gpflash_add_test_executable name)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    target_compile_options(${name} PRIVATE ${GPFLASH_WARNING_FLAGS})
    target_compile_definitions(${name} PRIVATE GPFLASH_DEBUG_CHECKS)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE GpFlash)
endfunction()

foreach(_test IN LISTS GPFLASH_TESTS)
    gpflash_add_test_executable(${_test})
    add_test(NAME ${_test} COMMAND ${_test})
endforeach()
# project()'s version, for the static_assert against Version.hpp.
target_compile_definitions(test_gpflash_version
                           PRIVATE GPFLASH_CMAKE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
                                   GPFLASH_CMAKE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
                                   GPFLASH_CMAKE_VERSION_PATCH=${PROJECT_VERSION_PATCH})

# A scan of the include graph rather than a compiled test, so it runs as a CMake script.
add_test(NAME check_prediction_only_includes
         COMMAND ${CMAKE_COMMAND} -DGPFLASH_INCLUDE_DIR=${GPFLASH_ROOT}/include
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_prediction_only_includes.cmake)

# Formatting, as a test so a ctest run enforces it. Another major version formats differently, so the check runs only
# on this one.
set(GPFLASH_CLANG_FORMAT_MAJOR 15)
find_program(GPFLASH_CLANG_FORMAT NAMES clang-format-${GPFLASH_CLANG_FORMAT_MAJOR} clang-format)
set(_clang_format_major "")
if(GPFLASH_CLANG_FORMAT)
    execute_process(COMMAND ${GPFLASH_CLANG_FORMAT} --version OUTPUT_VARIABLE _clang_format_version_text)
    string(REGEX MATCH "version ([0-9]+)" _ "${_clang_format_version_text}")
    set(_clang_format_major "${CMAKE_MATCH_1}")
endif()
if(_clang_format_major STREQUAL GPFLASH_CLANG_FORMAT_MAJOR)
    add_test(NAME check_formatting
             COMMAND ${CMAKE_COMMAND} -DGPFLASH_CLANG_FORMAT=${GPFLASH_CLANG_FORMAT} -DGPFLASH_ROOT=${GPFLASH_ROOT}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/check_formatting.cmake)
    # Reformats in place, with the same clang-format as the check.
    add_custom_target(clang-format-in-place-all
                      COMMAND ${CMAKE_COMMAND} -DGPFLASH_CLANG_FORMAT=${GPFLASH_CLANG_FORMAT}
                              -DGPFLASH_ROOT=${GPFLASH_ROOT} -DGPFLASH_FORMAT_IN_PLACE=ON
                              -P ${CMAKE_CURRENT_SOURCE_DIR}/check_formatting.cmake)
else()
    message(STATUS "GpFlash: no clang-format ${GPFLASH_CLANG_FORMAT_MAJOR}, so check_formatting is not registered.")
endif()

# Not required: only this test needs MPI, the library being built with GPFLASH_HAVE_MPI=0 (Platform.hpp). Absent
# MPI, the test is not registered rather than failing the configure. Build-tree paths, never exported.
option(GPFLASH_BUILD_MPI_TEST "Build and run ${GPFLASH_MPI_TEST} when MPI is found" ON)
set(GPFLASH_MPI_FOUND OFF)
if(GPFLASH_BUILD_MPI_TEST)
    # A plain lookup first, not FindMPI: that asks the MPI compiler wrapper for its flags, and a relocated MPI's
    # wrapper names paths on the machine that built it, which fails FindMPI's own test compile.
    find_library(GPFLASH_MPI_LIBRARY NAMES mpi)
    find_path(GPFLASH_MPI_INCLUDE_DIR mpi.h)
    find_program(GPFLASH_MPIEXEC NAMES mpiexec mpirun)
    if(GPFLASH_MPI_LIBRARY AND GPFLASH_MPI_INCLUDE_DIR AND GPFLASH_MPIEXEC)
        set(GPFLASH_MPI_FOUND ON)
        add_library(GpFlashTestMpi INTERFACE)
        target_include_directories(GpFlashTestMpi SYSTEM INTERFACE ${GPFLASH_MPI_INCLUDE_DIR})
        # Only the C API is used. Without these, Open MPI's and MPICH's mpi.h declare the deprecated C++ bindings, which
        # then fail to link when only libmpi is given.
        target_compile_definitions(GpFlashTestMpi INTERFACE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
        target_link_libraries(GpFlashTestMpi INTERFACE ${GPFLASH_MPI_LIBRARY})
        set(GPFLASH_MPI_LINK GpFlashTestMpi)
        set(GPFLASH_MPIEXEC_NUMPROC_FLAG -n)
    else()
        # Where the header sits under an MPI-specific directory, as on Debian.
        find_package(MPI COMPONENTS CXX QUIET)
        if(MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
            set(GPFLASH_MPI_FOUND ON)
            set(GPFLASH_MPI_LINK MPI::MPI_CXX)
            set(GPFLASH_MPIEXEC ${MPIEXEC_EXECUTABLE})
            set(GPFLASH_MPIEXEC_NUMPROC_FLAG ${MPIEXEC_NUMPROC_FLAG})
        endif()
    endif()
endif()

# Processes chosen so the split over the test's seven optimizations is uneven, which is the case worth covering.
# One thread per process: the assertions are about what the ranks agree on, not about speed, so more threads would
# only oversubscribe the host. TIMEOUT is the assertion for the failure cases, which hang rather than fail.
# PROCESSORS reserves a core per process, since ctest runs other tests beside this one.
set(GPFLASH_MPI_TEST_N_PROC 4)
set(GPFLASH_MPI_TEST_THREADS_PER_PROC 1)
set(GPFLASH_MPI_TEST_TIMEOUT_S 300)

if(GPFLASH_MPI_FOUND)
    gpflash_add_test_executable(${GPFLASH_MPI_TEST})
    target_compile_definitions(${GPFLASH_MPI_TEST} PRIVATE GPFLASH_HAVE_MPI=1)
    target_link_libraries(${GPFLASH_MPI_TEST} PRIVATE ${GPFLASH_MPI_LINK})

    add_test(NAME ${GPFLASH_MPI_TEST}
             COMMAND ${GPFLASH_MPIEXEC} ${GPFLASH_MPIEXEC_NUMPROC_FLAG} ${GPFLASH_MPI_TEST_N_PROC}
                     $<TARGET_FILE:${GPFLASH_MPI_TEST}>)
    set(_thread_env
        OMP_NUM_THREADS=${GPFLASH_MPI_TEST_THREADS_PER_PROC}
        OPENBLAS_NUM_THREADS=${GPFLASH_MPI_TEST_THREADS_PER_PROC})
    set_tests_properties(${GPFLASH_MPI_TEST} PROPERTIES
        TIMEOUT ${GPFLASH_MPI_TEST_TIMEOUT_S}
        PROCESSORS ${GPFLASH_MPI_TEST_N_PROC}
        ENVIRONMENT "${_thread_env}")
else()
    message(STATUS "GpFlash: no MPI library and launcher, or GPFLASH_BUILD_MPI_TEST is off, so ${GPFLASH_MPI_TEST} "
                   "is not registered.")
endif()
