// The version accessors against the macros they report, and the string against the numbers it is composed from.
#include <cstdio>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/Version.hpp"

// The build passes project()'s version, which CMakeLists.txt parses from this header; a parse that went wrong
// fails here.
static_assert(
        GPFLASH_VERSION_MAJOR == GPFLASH_CMAKE_VERSION_MAJOR && GPFLASH_VERSION_MINOR == GPFLASH_CMAKE_VERSION_MINOR
                && GPFLASH_VERSION_PATCH == GPFLASH_CMAKE_VERSION_PATCH,
        "CMakeLists.txt read a version from Version.hpp that differs from its macros");

using gpflash_test::kDetailLen;
using gpflash_test::report;

namespace {

bool test_version_accessors(const char* name) {
    const bool parts_ok = gpflash::version_major() == GPFLASH_VERSION_MAJOR
            && gpflash::version_minor() == GPFLASH_VERSION_MINOR && gpflash::version_patch() == GPFLASH_VERSION_PATCH;
    // The string must agree with the numbers it is composed from.
    char version_ref_str[kDetailLen];
    std::snprintf(
            version_ref_str,
            sizeof(version_ref_str),
            "%d.%d.%d",
            GPFLASH_VERSION_MAJOR,
            GPFLASH_VERSION_MINOR,
            GPFLASH_VERSION_PATCH);
    const bool consistent = std::string(gpflash::version_string()) == std::string(version_ref_str);
    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "%s  parts=%d consistent=%d",
            gpflash::version_string(),
            static_cast<int>(parts_ok),
            static_cast<int>(consistent));

    return report(name, parts_ok && consistent, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash version ==\n");
    bool ok = true;
    ok &= test_version_accessors("version_accessors");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
