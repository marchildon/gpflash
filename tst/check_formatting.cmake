# Fails if a C++ file under include/ or tst/ differs from what clang-format makes of it. With
# GPFLASH_FORMAT_IN_PLACE, reformats each file instead.
#
# clang-format finds the package's .clang-format by walking up from each file.
#
# Run as: cmake -DGPFLASH_CLANG_FORMAT=<clang-format> -DGPFLASH_ROOT=<package root> [-DGPFLASH_FORMAT_IN_PLACE=ON]
#         -P check_formatting.cmake
cmake_minimum_required(VERSION 3.22)

if(NOT EXISTS "${GPFLASH_ROOT}/.clang-format")
    message(FATAL_ERROR "GPFLASH_ROOT='${GPFLASH_ROOT}' holds no .clang-format.")
endif()

# Relative to the package, so a build* directory above it does not match the filter below.
file(GLOB_RECURSE relative_source_list RELATIVE "${GPFLASH_ROOT}"
     "${GPFLASH_ROOT}/include/*.hpp" "${GPFLASH_ROOT}/tst/*.hpp" "${GPFLASH_ROOT}/tst/*.cpp")
# A build tree configured inside the package holds generated sources that are not ours to format.
list(FILTER relative_source_list EXCLUDE REGEX "(^|/)build[^/]*/")

set(unformatted_message "")
foreach(relative_source IN LISTS relative_source_list)
    set(source "${GPFLASH_ROOT}/${relative_source}")
    if(GPFLASH_FORMAT_IN_PLACE)
        execute_process(
            COMMAND "${GPFLASH_CLANG_FORMAT}" --style=file -i "${source}"
            RESULT_VARIABLE format_result)
        if(NOT format_result EQUAL 0)
            message(FATAL_ERROR "${GPFLASH_CLANG_FORMAT} failed on ${source}.")
        endif()
        continue()
    endif()

    execute_process(
        COMMAND "${GPFLASH_CLANG_FORMAT}" --style=file "${source}"
        OUTPUT_VARIABLE formatted_text
        RESULT_VARIABLE format_result)
    if(NOT format_result EQUAL 0)
        message(FATAL_ERROR "${GPFLASH_CLANG_FORMAT} failed on ${source}.")
    endif()
    file(READ "${source}" source_text)
    if(NOT formatted_text STREQUAL source_text)
        string(APPEND unformatted_message "\n  ${relative_source}")
    endif()
endforeach()

list(LENGTH relative_source_list n_source)
if(GPFLASH_FORMAT_IN_PLACE)
    message(STATUS "Reformatted ${n_source} files.")
elseif(unformatted_message)
    message(FATAL_ERROR "Not formatted (build the clang-format-in-place-all target):${unformatted_message}")
else()
    message(STATUS "All ${n_source} files are formatted.")
endif()
