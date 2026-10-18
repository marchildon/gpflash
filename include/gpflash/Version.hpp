/**
 * Version.hpp -- library version. The macros are for compile-time #if checks in consumer code; the functions return
 * the same version at run time. The library is header-only, so both report the headers the caller compiled
 * against.
 *
 * The one place the version is set: CMakeLists.txt and pyproject.toml read the three macros from here.
 */
#pragma once

#include "gpflash/core/Platform.hpp"  // GPFLASH_STR

#define GPFLASH_VERSION_MAJOR 0
#define GPFLASH_VERSION_MINOR 3
#define GPFLASH_VERSION_PATCH 0

namespace gpflash {

inline int version_major() { return GPFLASH_VERSION_MAJOR; }
inline int version_minor() { return GPFLASH_VERSION_MINOR; }
inline int version_patch() { return GPFLASH_VERSION_PATCH; }

// Composed rather than a fourth macro, so the string cannot disagree with the three numbers above.
inline const char* version_string() {
    return GPFLASH_STR(GPFLASH_VERSION_MAJOR) "." GPFLASH_STR(GPFLASH_VERSION_MINOR) "." GPFLASH_STR(
            GPFLASH_VERSION_PATCH);
}

}  // namespace gpflash
