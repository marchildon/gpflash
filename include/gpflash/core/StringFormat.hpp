/**
 * StringFormat.hpp -- printf-style formatting into a std::string sized to fit, so no caller sizes a buffer.
 */
#pragma once

#include <cstdarg>    // va_list
#include <cstddef>    // std::size_t
#include <cstdio>     // std::vsnprintf
#include <stdexcept>  // std::runtime_error, for an output error
#include <string>     // std::string

#include "gpflash/core/Platform.hpp"  // GPFLASH_PRINTF_FORMAT

namespace gpflash {
namespace detail {

// std::snprintf's output, whatever its length.
GPFLASH_PRINTF_FORMAT(1, 2) inline std::string to_formatted_string(const char* format, ...) {
    // The arguments are read twice, once to measure and once to write, so the list is copied first.
    va_list arg_list;
    va_start(arg_list, format);
    va_list arg_list_copy;
    va_copy(arg_list_copy, arg_list);
    const int n_char = std::vsnprintf(nullptr, 0, format, arg_list);
    va_end(arg_list);
    // A negative count is an output error. The copy still needs its va_end, so it is ended before the throw.
    if (n_char < 0) {
        va_end(arg_list_copy);
        throw std::runtime_error("gpflash::detail::to_formatted_string: vsnprintf reported an output error.");
    }

    // One more for the terminating null vsnprintf writes, then dropped.
    std::string formatted_str(static_cast<std::size_t>(n_char) + 1, '\0');
    std::vsnprintf(&formatted_str[0], formatted_str.size(), format, arg_list_copy);
    va_end(arg_list_copy);
    formatted_str.resize(static_cast<std::size_t>(n_char));

    return formatted_str;
}

}  // namespace detail
}  // namespace gpflash
