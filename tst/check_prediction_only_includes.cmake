# Fails if a prediction-only header reaches posterior_build/, training/ or optimization/.
#
# Every header exists on disk and the package compiles either way, so nothing else catches a stray include.
#
# Run as: cmake -DGPFLASH_INCLUDE_DIR=<include dir> -P check_prediction_only_includes.cmake
cmake_minimum_required(VERSION 3.22)

if(NOT IS_DIRECTORY "${GPFLASH_INCLUDE_DIR}/gpflash")
    message(FATAL_ERROR "GPFLASH_INCLUDE_DIR='${GPFLASH_INCLUDE_DIR}' holds no gpflash/ directory.")
endif()

# The README's two configurations, by directory under gpflash/.
set(prediction_only_dir_list
    core inputs linear_algebra preconditioning kernels mean assembly posterior_cache posterior_eval)
set(full_build_only_dir_list posterior_build training optimization)
list(JOIN full_build_only_dir_list "|" full_build_only_dir_alternation)
set(full_build_only_regex "^gpflash/(${full_build_only_dir_alternation})/")

# A directory in neither list would be neither a root nor a leak, so the check below would skip it unreported.
file(GLOB gpflash_entry_list RELATIVE "${GPFLASH_INCLUDE_DIR}/gpflash" "${GPFLASH_INCLUDE_DIR}/gpflash/*")
foreach(gpflash_entry IN LISTS gpflash_entry_list)
    if(IS_DIRECTORY "${GPFLASH_INCLUDE_DIR}/gpflash/${gpflash_entry}"
       AND NOT gpflash_entry IN_LIST prediction_only_dir_list
       AND NOT gpflash_entry IN_LIST full_build_only_dir_list)
        message(FATAL_ERROR "gpflash/${gpflash_entry}/ is in neither prediction_only_dir_list nor "
                            "full_build_only_dir_list; add it to one.")
    endif()
endforeach()

# A consumer of the prediction-only configuration may include any of its headers directly, so each is a root: the
# headers directly under gpflash/, which belong to no directory, and every header in a prediction-only directory.
file(GLOB root_header_list RELATIVE "${GPFLASH_INCLUDE_DIR}" "${GPFLASH_INCLUDE_DIR}/gpflash/*.hpp")
foreach(root_dir IN LISTS prediction_only_dir_list)
    file(GLOB_RECURSE root_dir_header_list
         RELATIVE "${GPFLASH_INCLUDE_DIR}" "${GPFLASH_INCLUDE_DIR}/gpflash/${root_dir}/*.hpp")
    list(APPEND root_header_list ${root_dir_header_list})
endforeach()
# Both delimiters, since the quoted and the angle-bracket form each resolve through the include directory.
set(include_regex "^[ \t]*#[ \t]*include[ \t]*([<\"])([^\">]+)[\">]")

# Breadth-first over the include graph. pulled_in_by_<header> records the includer, so a failure names the edge.
set(reached_header_list ${root_header_list})
set(queue_header_list ${root_header_list})
while(queue_header_list)
    list(POP_FRONT queue_header_list header)
    if(NOT EXISTS "${GPFLASH_INCLUDE_DIR}/${header}")
        continue()
    endif()

    get_filename_component(header_dir "${header}" DIRECTORY)
    file(STRINGS "${GPFLASH_INCLUDE_DIR}/${header}" include_line_list REGEX "${include_regex}")
    foreach(include_line IN LISTS include_line_list)
        string(REGEX MATCH "${include_regex}" include_match "${include_line}")
        set(include_delimiter "${CMAKE_MATCH_1}")
        set(included_header "${CMAKE_MATCH_2}")
        if(NOT included_header MATCHES "^gpflash/")
            # A quoted path found beside the includer compiles, but the walk keys headers by their gpflash/ spelling,
            # so it would follow this one under a second name.
            if(include_delimiter STREQUAL "\"" AND EXISTS "${GPFLASH_INCLUDE_DIR}/${header_dir}/${included_header}")
                message(FATAL_ERROR "${header} includes \"${included_header}\" relative to itself; spell it from "
                                    "gpflash/ so this check can follow it.")
            endif()
            continue()
        endif()
        if(NOT included_header IN_LIST reached_header_list)
            list(APPEND reached_header_list "${included_header}")
            list(APPEND queue_header_list "${included_header}")
            set("pulled_in_by_${included_header}" "${header}")
        endif()
    endforeach()
endwhile()

set(leak_message "")
foreach(header IN LISTS reached_header_list)
    if(header MATCHES "${full_build_only_regex}")
        string(APPEND leak_message "\n  ${header}  <-- included by ${pulled_in_by_${header}}")
    endif()
endforeach()

list(LENGTH reached_header_list n_reached_header)
if(leak_message)
    message(FATAL_ERROR "The prediction-only configuration reaches full-build-only headers:${leak_message}")
endif()
message(STATUS "The prediction-only configuration reaches ${n_reached_header} headers, none full-build-only.")
